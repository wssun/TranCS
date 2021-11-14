import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)
from modules import SeqEncoder, BOWEncoder, SeqEncoder2


class JointEmbeder(nn.Module):
    def __init__(self, config):
        super(JointEmbeder, self).__init__()
        self.conf = config
        self.margin = config['margin']
        self.dropout = config['dropout']
        self.n_hidden = config['n_hidden']

        self.name_encoder = SeqEncoder(config['n_words'], config['emb_size'], config['lstm_dims'])
        self.tok_encoder = BOWEncoder(config['n_words'], config['emb_size'], config['n_hidden'])
        self.api_encoder = SeqEncoder(config['n_words'], config['emb_size'], config['lstm_dims'])
        self.desc_encoder = SeqEncoder2(config['n_words'], config['emb_size'], config['n_hidden'])

        self.w_name = nn.Linear(2 * config['lstm_dims'], config['n_hidden'])
        self.w_tok = nn.Linear(config['emb_size'], config['n_hidden'])
        self.w_api = nn.Linear(2 * config['lstm_dims'], config['n_hidden'])
        self.fuse3 = nn.Linear(config['n_hidden'], config['n_hidden'])

        self.init_weights()

    def init_weights(self):
        for m in [self.w_name, self.w_tok, self.fuse3]:
            m.weight.data.uniform_(-0.1, 0.1)
            nn.init.constant_(m.bias, 0.)

    def code_encoding(self, name, name_len, tokens, tok_len, api, api_len):
        name_repr = self.name_encoder(name, name_len)
        tok_repr = self.tok_encoder(tokens, tok_len)
        api_repr = self.api_encoder(api, api_len)
        code_repr = self.fuse3(torch.tanh(self.w_name(name_repr) + self.w_tok(tok_repr) + self.w_api(api_repr)))
        return code_repr

    def desc_encoding(self, desc, desc_len):
        batch_size = desc.size()[0]
        desc_enc_hidden = self.desc_encoder.init_hidden(batch_size)
        desc_feat, desc_enc_hidden = self.desc_encoder(desc, desc_len, desc_enc_hidden)
        desc_enc_hidden = desc_enc_hidden[0]

        return desc_enc_hidden

    def similarity(self, code_vec, desc_vec):
        assert self.conf['sim_measure'] in ['cos', 'poly', 'euc', 'sigmoid', 'gesd',
                                            'aesd'], "invalid similarity measure"
        if self.conf['sim_measure'] == 'cos':
            return F.cosine_similarity(code_vec, desc_vec)
        elif self.conf['sim_measure'] == 'poly':
            return (0.5 * torch.matmul(code_vec, desc_vec.t()).diag() + 1) ** 2
        elif self.conf['sim_measure'] == 'sigmoid':
            return torch.tanh(torch.matmul(code_vec, desc_vec.t()).diag() + 1)
        elif self.conf['sim_measure'] in ['euc', 'gesd', 'aesd']:
            euc_dist = torch.dist(code_vec, desc_vec, 2)
            euc_sim = 1 / (1 + euc_dist)
            if self.conf['sim_measure'] == 'euc': return euc_sim
            sigmoid_sim = torch.sigmoid(torch.matmul(code_vec, desc_vec.t()).diag() + 1)
            if self.conf['sim_measure'] == 'gesd':
                return euc_sim * sigmoid_sim
            elif self.conf['sim_measure'] == 'aesd':
                return 0.5 * (euc_sim + sigmoid_sim)

    def forward(self, name, name_len, tokens, tok_len, api, api_len,
                desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):

        code_repr = self.code_encoding(name, name_len, tokens, tok_len, api, api_len)

        desc_anchor_repr = self.desc_encoding(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encoding(desc_neg, desc_neg_len)

        anchor_sim = self.similarity(code_repr, desc_anchor_repr)
        neg_sim = self.similarity(code_repr, desc_neg_repr)

        loss = (self.margin - anchor_sim + neg_sim).clamp(min=1e-6).mean()

        return loss
