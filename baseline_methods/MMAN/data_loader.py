import sys
import torch
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
import collections
import dgl
import pandas as pd

import pickle

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import configs
from util_cfg import get_one_cfg_npy_info
from util_ast import build_tree

PAD_ID, UNK_ID = [0, 1]

ASTBatch = collections.namedtuple('ASTBatch', ['graph', 'mask', 'wordid', 'label'])


def batcher(device):
    def batcher_dev(batch):
        tokens, tok_len, tree, tree_node_num, init_input, adjmat, node_mask, good_desc, good_desc_len, bad_desc, bad_desc_len = zip(
            *batch)

        tree = dgl.batch(tree)
        tree = ASTBatch(graph=tree,
                        mask=tree.ndata['mask'].to(device),
                        wordid=tree.ndata['x'].to(device),
                        label=tree.ndata['y'].to(device))

        tokens = tuplelist2tensor_long(tokens).to(device)
        tok_len = tuple2tensor_long(tok_len).to(device)

        tree_node_num = tuple2tensor_long(tree_node_num).to(device)
        init_input = tuple3list2tensor_float(init_input).to(device)
        adjmat = tuple3list2tensor_float(adjmat).to(device)
        node_mask = tuplelist2tensor_long(node_mask).to(device)

        good_desc = tuplelist2tensor_long(good_desc).to(device)
        good_desc_len = tuple2tensor_long(good_desc_len).to(device)
        bad_desc = tuplelist2tensor_long(bad_desc).to(device)
        bad_desc_len = tuple2tensor_long(bad_desc_len).to(device)

        return tokens, tok_len, tree, tree_node_num, init_input, adjmat, node_mask, good_desc, good_desc_len, bad_desc, bad_desc_len

    return batcher_dev


class CodeSearchDataset(data.Dataset):
    def __init__(self, config, data_dir, f_token, max_tok_len, f_ast, f_ast_dict, f_cfg, max_node_num, f_desc,
                 max_desc_len):
        self.max_tok_len = max_tok_len
        self.max_node_num = max_node_num
        self.max_desc_len = max_desc_len
        self.max_word_num = config['max_word_num']

        self.n_edge_types = config['n_edge_types']
        self.state_dim = config['state_dim']
        self.annotation_dim = config['annotation_dim']

        self.trees = []
        self.trees_num = []

        print("Loading Data...")

        # ast_tree_json = json.loads(open(data_dir+f_ast, 'r').readline())
        # ast_tree_jsons = pd.read_pickle(data_dir + f_ast)
        ast_tree_jsons = pickle.load(open(data_dir + f_ast, 'rb'))

        vacab_ast_dict = json.loads(open(data_dir + f_ast_dict, 'r').readline())

        for i, tree_json in enumerate(ast_tree_jsons):
            self.trees.append(build_tree(tree_json, vacab_ast_dict))
            self.trees_num.append(self.trees[i].number_of_nodes())

        self.json_dict = json.loads(open(data_dir + f_cfg, 'r').readline())

        table_tokens = tables.open_file(data_dir + f_token)
        self.tokens = table_tokens.get_node('/phrases')[:].astype(np.long)
        self.idx_tokens = table_tokens.get_node('/indices')[:]

        table_desc = tables.open_file(data_dir + f_desc)
        self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
        self.idx_descs = table_desc.get_node('/indices')[:]

        assert self.idx_tokens.shape[0] == self.idx_descs.shape[0]
        assert self.idx_tokens.shape[0] == len(self.trees)

        self.data_len = self.idx_descs.shape[0]
        print("{} entries".format(self.data_len))

    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = np.append(seq, [PAD_ID] * (maxlen - len(seq)))
        seq = seq[:maxlen]
        return seq

    def __getitem__(self, offset):

        tree = self.trees[offset]
        tree_node_num = self.trees_num[offset]

        adjmat, init_input, node_mask = get_one_cfg_npy_info(self.json_dict[str(offset)],
                                                             self.max_node_num, self.n_edge_types, self.state_dim,
                                                             self.max_word_num)

        len, pos = self.idx_tokens[offset][0], self.idx_tokens[offset][1]
        tok_len = min(int(len), self.max_tok_len)
        tokens = self.tokens[pos:pos + tok_len]
        tokens = self.pad_seq(tokens, self.max_tok_len)

        len, pos = self.idx_descs[offset][0], self.idx_descs[offset][1]
        good_desc_len = min(int(len), self.max_desc_len)
        good_desc = self.descs[pos: pos + good_desc_len]
        good_desc = self.pad_seq(good_desc, self.max_desc_len)

        rand_offset = random.randint(0, self.data_len - 1)
        len, pos = self.idx_descs[rand_offset][0], self.idx_descs[rand_offset][1]
        bad_desc_len = min(int(len), self.max_desc_len)
        bad_desc = self.descs[pos: pos + bad_desc_len]
        bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

        return tokens, tok_len, tree, tree_node_num, init_input, adjmat, node_mask, good_desc, good_desc_len, bad_desc, bad_desc_len

    def __len__(self):
        return self.data_len

def tuple2tensor_long(long_tuple):
    long_numpy = np.zeros(len(long_tuple))
    for index, value in enumerate(long_tuple):
        long_numpy[index] = value
    long_tensor = torch.from_numpy(long_numpy).type(torch.LongTensor)
    return long_tensor

def tuplelist2tensor_long(long_tuple_list):
    long_numpy = np.zeros([len(long_tuple_list), len(long_tuple_list[0])])
    for index, value in enumerate(long_tuple_list):
        long_numpy[index] = value
    long_tensor = torch.from_numpy(long_numpy).type(torch.LongTensor)
    return long_tensor

def tuple3list2tensor_float(float_tuple_3list):
    float_numpy = np.zeros([len(float_tuple_3list), float_tuple_3list[0].shape[0], float_tuple_3list[0].shape[1]])
    for index, value in enumerate(float_tuple_3list):
        float_numpy[index] = value
    float_tensor = torch.from_numpy(float_numpy).float()
    return float_tensor

