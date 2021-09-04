import os
import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqEncoder_LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers, embedding):
        super(SeqEncoder_LSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = embedding

        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0, batch_first=True, bidirectional=False)

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
