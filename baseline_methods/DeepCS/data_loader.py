import torch.utils.data as data
import tables
import random
import numpy as np
from utils import PAD_ID, UNK_ID, indexes2sent

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class CodeSearchDataset(data.Dataset):

    def __init__(self, data_dir, f_name=None, max_name_len=None, f_tokens=None, max_tok_len=None, f_api=None,
                 max_api_len=None, f_descs=None, max_desc_len=None, is_train=True):
        self.is_train = is_train

        self.max_name_len = max_name_len
        self.max_tok_len = max_tok_len
        self.max_api_len = max_api_len
        self.max_desc_len = max_desc_len

        print("loading data...")

        table_name = tables.open_file(data_dir + f_name)
        self.names = table_name.get_node('/phrases')[:].astype(np.long)
        self.idx_names = table_name.get_node('/indices')[:]

        table_tokens = tables.open_file(data_dir + f_tokens)
        self.tokens = table_tokens.get_node('/phrases')[:].astype(np.long)
        self.idx_tokens = table_tokens.get_node('/indices')[:]

        table_api = tables.open_file(data_dir + f_api)
        self.api = table_api.get_node('/phrases')[:].astype(np.long)
        self.idx_api = table_api.get_node('/indices')[:]

        table_desc = tables.open_file(data_dir + f_descs)
        self.descs = table_desc.get_node('/phrases')[:].astype(np.long)
        self.idx_descs = table_desc.get_node('/indices')[:]

        assert self.idx_names.shape[0] == self.idx_tokens.shape[0]
        assert self.idx_tokens.shape[0] == self.idx_api.shape[0]
        assert self.idx_api.shape[0] == self.idx_descs.shape[0]

        self.data_len = self.idx_descs.shape[0]

        print("{} entries".format(self.data_len))

    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = np.append(seq, [PAD_ID] * (maxlen - len(seq)))
        seq = seq[:maxlen]
        return seq

    def __getitem__(self, offset):
        len, pos = self.idx_names[offset][0], self.idx_names[offset][1]
        name_len = min(int(len), self.max_name_len)
        name = self.names[pos: pos + name_len]
        name = self.pad_seq(name, self.max_name_len)

        len, pos = self.idx_tokens[offset][0], self.idx_tokens[offset][1]
        tok_len = min(int(len), self.max_tok_len)
        tokens = self.tokens[pos:pos + tok_len]
        tokens = self.pad_seq(tokens, self.max_tok_len)

        len, pos = self.idx_api[offset][0], self.idx_api[offset][1]
        api_len = min(int(len), self.max_api_len)
        api = self.api[pos:pos + api_len]
        api = self.pad_seq(api, self.max_api_len)

        len, pos = self.idx_descs[offset][0], self.idx_descs[offset][1]
        good_desc_len = min(int(len), self.max_desc_len)
        good_desc = self.descs[pos:pos + good_desc_len]
        good_desc = self.pad_seq(good_desc, self.max_desc_len)

        rand_offset = random.randint(0, self.data_len - 1)
        len, pos = self.idx_descs[rand_offset][0], self.idx_descs[rand_offset][1]
        bad_desc_len = min(int(len), self.max_desc_len)
        bad_desc = self.descs[pos:pos + bad_desc_len]
        bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

        if self.is_train:
            return name, name_len, tokens, tok_len, api, api_len, good_desc, good_desc_len, bad_desc, bad_desc_len
        else:
            return name, name_len, tokens, tok_len, api, api_len, good_desc, good_desc_len

    def __len__(self):
        return self.data_len
