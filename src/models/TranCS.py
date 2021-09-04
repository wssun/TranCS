import torch
import torch.nn as nn
import torch.nn.functional as F

from .Modules import SeqEncoder_LSTM


class TranEmbeder(nn.Module):
    def __init__(self, config):
        super(TranEmbeder, self).__init__()

        self.conf = config

        self.n_tran_words = config['n_tran_words']
        self.n_doc_words = config['n_doc_words']
        self.n_tran_doc_words = config['n_tran_doc_words']

        self.code_nn = config['code_nn']
        self.doc_nn = config['doc_nn']
        self.mode = config['mode']

        self.margin = config['margin']
        self.dropout = config['dropout']

        self.emb_size = config['emb_size']
        self.n_hidden = config['n_hidden']
        self.n_layers_LSTM = config['n_layers_LSTM']
        self.doc_n_layers = config['n_layers_LSTM']

        self.tran_with_attention = config['tran_with_attention']
        self.doc_with_attention = config['doc_with_attention']
        self.tran_transform = config['tran_transform']
        self.doc_transform = config['doc_transform']

        self.transform_every_modal = config['transform_every_modal']
        self.transform_attn_out = config['transform_attn_out']

        self.use_tanh = config['use_tanh']

        if self.transform_every_modal:
            self.linear_single_modal = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                                     nn.Tanh(),
                                                     nn.Linear(self.n_hidden, self.n_hidden))
        if self.transform_attn_out:
            self.linear_attn_out = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                                 nn.Tanh(),
                                                 nn.Linear(self.n_hidden, self.n_hidden))

        self.tran_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.tran_attn_scalar = nn.Linear(self.n_hidden, 1)

        self.doc_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.doc_attn_scalar = nn.Linear(self.n_hidden, 1)

        self.embedding = nn.Embedding(self.n_tran_doc_words, self.emb_size, padding_idx=0)
        self.init_xavier_linear(self.embedding, init_bias=False)

        self.init_encoder()

    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)

    def init_encoder(self):
        self.tran_encoder = SeqEncoder_LSTM(self.n_tran_words, self.emb_size, self.n_hidden,
                                            self.n_layers_LSTM, self.embedding)

        self.doc_encoder = SeqEncoder_LSTM(self.n_doc_words, self.emb_size, self.n_hidden, self.doc_n_layers,
                                           self.embedding)

    def LSTM_encoding(self, tran, tran_len):
        batch_size = tran.size()[0]
        tran_enc_hidden = self.tran_encoder.init_hidden(batch_size)
        tran_feat, tran_enc_hidden = self.tran_encoder(tran, tran_len, tran_enc_hidden)
        tran_enc_hidden = tran_enc_hidden[0]

        if self.conf['transform_every_modal']:
            tran_enc_hidden = torch.tanh(
                self.linear_single_modal(
                    F.dropout(tran_enc_hidden, self.dropout, training=self.training)
                )
            )
        elif self.conf['use_tanh']:
            tran_enc_hidden = torch.tanh(tran_enc_hidden)

        if self.conf['tran_with_attention']:
            seq_len = tran_feat.size()[1]

            device = torch.device(f"cuda:{self.conf['gpu_id']}" if torch.cuda.is_available() else "cpu")
            unpack_len_list = tran_len.long().to(device)
            range_tensor = torch.arange(seq_len).to(device)
            mask_1forgt0 = range_tensor[None, :] < unpack_len_list[:, None]
            mask_1forgt0 = mask_1forgt0.reshape(-1, seq_len)

            tran_sa_tanh = torch.tanh(
                self.tran_attn(tran_feat.reshape(-1, self.n_hidden)))
            tran_sa_tanh = F.dropout(tran_sa_tanh, self.dropout, training=self.training)
            tran_sa_tanh = self.tran_attn_scalar(tran_sa_tanh).reshape(-1, seq_len)
            tran_feat = tran_feat.reshape(-1, seq_len, self.n_hidden)

            self_attn_tran_feat = None
            for _i in range(batch_size):
                tran_sa_tanh_one = torch.masked_select(tran_sa_tanh[_i, :], mask_1forgt0[_i, :]).reshape(1,
                                                                                                         -1)
                attn_w_one = F.softmax(tran_sa_tanh_one, dim=1).reshape(1, 1, -1)

                attn_feat_one = torch.masked_select(tran_feat[_i, :, :].reshape(1, seq_len, self.n_hidden),
                                                    mask_1forgt0[_i, :].reshape(1, seq_len, 1)).reshape(1, -1,
                                                                                                        self.n_hidden)
                out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.n_hidden)
                self_attn_tran_feat = out_to_cat if self_attn_tran_feat is None else torch.cat(
                    (self_attn_tran_feat, out_to_cat), 0)
        else:
            self_attn_tran_feat = tran_enc_hidden.reshape(batch_size, self.n_hidden)

        if self.conf['transform_attn_out']:
            self_attn_tran_feat = torch.tanh(
                self.linear_attn_out(
                    F.dropout(self_attn_tran_feat, self.dropout, training=self.training)
                )
            )
        elif self.conf['use_tanh']:
            self_attn_tran_feat = torch.tanh(self_attn_tran_feat)

        return self_attn_tran_feat

    def tran_sequence_encoding(self, tran, tran_len):
        output = self.LSTM_encoding(tran, tran_len)
        return output

    def tran_block_encoding(self, tran, tran_block_len):
        batch_size = tran.size()[0]
        output_list = []
        for i in range(batch_size):
            output = self.LSTM_encoding(tran[i], tran_block_len[i])
            output_list.append(output)
        output = torch.stack(output_list)
        output = F.max_pool2d(output, kernel_size=(self.conf['tran_block_len'], 1), stride=1).squeeze(1)
        return output

    def code_encoding(self, tran, tran_len, tran_block_len):
        return self.tran_block_encoding(tran, tran_block_len)

    def doc_encoding(self, doc, doc_len):
        batch_size = doc.size()[0]
        doc_enc_hidden = self.doc_encoder.init_hidden(batch_size)
        doc_output, doc_hidden = self.doc_encoder(doc, doc_len, doc_enc_hidden)

        doc_hidden = doc_hidden[0]

        if doc_hidden.size()[0] == 1:
            doc_hidden = doc_hidden.reshape(doc_hidden.size()[1], doc_enc_hidden.size()[2])

        if self.transform_every_modal:
            doc_hidden = torch.tanh(
                self.linear_single_modal(F.dropout(doc_hidden, self.dropout, training=self.training)))
        elif self.use_tanh:
            doc_hidden = torch.tanh(doc_hidden)

        if self.doc_with_attention:
            seq_len = doc_output.size()[1]

            device = torch.device(f"cuda:{self.conf['gpu_id']}" if torch.cuda.is_available() else "cpu")
            unpack_len_list = doc_len.long().to(device)
            range_tensor = torch.arange(seq_len).to(device)
            mask_1forgt0 = range_tensor[None, :] < unpack_len_list[:, None]
            mask_1forgt0 = mask_1forgt0.reshape(-1, seq_len)

            doc_sa_tanh = torch.tanh(
                self.doc_attn(doc_output.reshape(-1, self.n_hidden)))
            doc_sa_tanh = F.dropout(doc_sa_tanh, self.dropout, training=self.training)
            doc_sa_tanh = self.doc_attn_scalar(doc_sa_tanh).reshape(-1, seq_len)
            doc_output = doc_output.reshape(-1, seq_len, self.n_hidden)

            self_attn_doc_feat = None
            for _i in range(batch_size):
                doc_sa_tanh_one = torch.masked_select(doc_sa_tanh[_i, :], mask_1forgt0[_i, :]).reshape(1, -1)
                attn_w_one = F.softmax(doc_sa_tanh_one, dim=1).reshape(1, 1, -1)

                attn_feat_one = torch.masked_select(doc_output[_i, :, :].reshape(1, seq_len, self.n_hidden),
                                                    mask_1forgt0[_i, :].reshape(1, seq_len, 1)).reshape(1, -1,
                                                                                                        self.n_hidden)
                out_to_cat = torch.bmm(attn_w_one, attn_feat_one).reshape(1, self.n_hidden)
                self_attn_doc_feat = out_to_cat if self_attn_doc_feat is None else torch.cat(
                    (self_attn_doc_feat, out_to_cat), 0)
        else:
            self_attn_doc_feat = doc_hidden.reshape(batch_size, self.n_hidden)

        if self.transform_attn_out:
            self_attn_doc_feat = torch.tanh(
                self.linear_attn_out(
                    F.dropout(self_attn_doc_feat, self.opt.dropout, training=self.training)))

        return self_attn_doc_feat

    def forward(self, tran, tran_len, tran_block_len, doc_anchor, doc_anchor_len, doc_neg, doc_neg_len):
        code_repr = self.code_encoding(tran, tran_len, tran_block_len)

        doc_anchor_repr = self.doc_encoding(doc_anchor, doc_anchor_len)
        doc_neg_repr = self.doc_encoding(doc_neg, doc_neg_len)

        anchor_sim = F.cosine_similarity(code_repr, doc_anchor_repr)
        neg_sim = F.cosine_similarity(code_repr, doc_neg_repr)

        loss = (self.margin - anchor_sim + neg_sim).clamp(min=1e-6).mean()

        return loss
