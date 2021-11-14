import os
import numpy as np
import math
import dgl

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.init_xavier_linear(self.embedding, init_bias=False)
        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0.1, batch_first=True, bidirectional=False)

    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().requires_grad_(),
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().requires_grad_())

    def forward(self, inputs, input_lens=None, hidden=None):
        batch_size, seq_len = inputs.size()
        inputs = self.embedding(inputs)

        if input_lens is not None:
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)

        hids, (h_n, c_n) = self.lstm(inputs, hidden)

        if input_lens is not None:
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
            c_n = c_n.index_select(1, inv_indices)

        h_n = h_n[0]
        c_n = c_n[0]

        return hids, (h_n, c_n)


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].reshape(nodes.mailbox['h'].size(0), -1)
        f = torch.sigmoid(self.U_f(h_cat)).reshape(*nodes.mailbox['h'].size())
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return_iou = self.U_iou(h_cat)

        return {'iou': return_iou, 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)

        return {'h': h, 'c': c}


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):
    def __init__(self, config):
        super(TreeLSTM, self).__init__()

        self.emb_size = config['emb_size']
        self.hidden_size = config['n_hidden']

        self.vocab_size = config['n_ast_words']
        self.treelstm_cell_type = config['treelstm_cell_type']

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size, padding_idx=0)

        self.init_xavier_linear(self.embedding, init_bias=False)

        cell = TreeLSTMCell if self.treelstm_cell_type == 'nary' else ChildSumTreeLSTMCell
        self.cell = cell(self.emb_size, self.hidden_size)

    def forward(self, batch, enc_hidden):

        g = batch.graph
        # g.update_all(self.cell.message_func, self.cell.reduce_func)
        # g.update_all(self.cell.reduce_func)
        # g.apply_nodes(self.cell.apply_node_func)
        g.register_message_func(self.cell.message_func)
        g.register_reduce_func(self.cell.reduce_func)
        g.register_apply_node_func(self.cell.apply_node_func)

        g.ndata['h'] = enc_hidden[0]
        g.ndata['c'] = enc_hidden[1]
        embeds = self.embedding(batch.wordid * batch.mask)
        g.ndata['iou'] = self.cell.W_iou(embeds) * batch.mask.float().unsqueeze(-1)

        dgl.prop_nodes_topo(g)

        all_node_h_in_batch = g.ndata.pop('h')
        all_node_c_in_batch = g.ndata.pop('c')

        return all_node_h_in_batch, all_node_c_in_batch

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        return (weight.new(batch_size, self.hidden_size).zero_().requires_grad_(),
                weight.new(batch_size, self.hidden_size).zero_().requires_grad_())

    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)


class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    def __init__(self, state_dim, n_node, n_edge_types):
        super(Propogator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_in, state_out, state_cur, A):
        A_in = A[:, :, :self.n_node * self.n_edge_types]
        A_out = A[:, :, self.n_node * self.n_edge_types:]

        a_in = torch.bmm(A_in, state_in)
        a_out = torch.bmm(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    def __init__(self, config):
        super(GGNN, self).__init__()

        assert (config['state_dim'] >= config['annotation_dim'], \
                'state_dim must be no less than annotation_dim')

        self.config = config
        self.n_layers = config['n_layers']
        self.n_hidden = config['n_hidden']

        self.state_dim = config['state_dim']

        self.n_edge_types = config['n_edge_types']
        self.n_node = config['n_node']
        self.n_steps = config['n_steps']
        self.output_type = config['output_type']
        self.batch_size = config['batch_size']

        for i in range(self.n_edge_types):
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        self.propogator = Propogator(self.state_dim, self.n_node, self.n_edge_types)

        self.out_mlp = nn.Sequential(
            nn.Dropout(p=config['dropout'], inplace=False),
            nn.Linear(self.state_dim + self.state_dim, self.state_dim),
            nn.Tanh(),
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self.init_xavier_linear(m)

    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)

    def forward(self, prop_state, A, node_mask):

        initial_prop_state = prop_state

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []

            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.reshape(-1, self.n_node * self.n_edge_types, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.reshape(-1, self.n_node * self.n_edge_types, self.state_dim)

            prop_state = self.propogator(in_states, out_states, prop_state, A)

        join_state = torch.cat((prop_state, initial_prop_state), 2)
        output = self.out_mlp(join_state)

        return output


from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_word_weights(vocab_size, padding_idx=0):
    def cal_weight(word_idx):
        return 1 - math.exp(-word_idx)

    weight_table = np.array([cal_weight(w) for w in range(vocab_size)])
    if padding_idx is not None:
        weight_table[padding_idx] = 0.
    return torch.FloatTensor(weight_table)
