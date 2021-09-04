import torch.utils.data as data
import tables
import random
import numpy as np

from dataset_utils import PAD_ID, UNK_ID, EOS_ID, pre_block


class TranCSDataset(data.Dataset):

    def __init__(self, config, data_dir, is_train, f_trans, max_tran_len, max_tran_seq_len,
                 max_tran_block_len, f_docs, max_doc_len):
        self.conf = config

        self.is_train = is_train

        self.max_tran_len = max_tran_len
        self.max_tran_seq_len = max_tran_seq_len
        self.max_tran_block_len = max_tran_block_len
        self.max_doc_len = max_doc_len

        table_tran = tables.open_file(data_dir + f_trans)
        self.trans = table_tran.get_node('/phrases')[:].astype(np.int32)
        self.idx_trans = table_tran.get_node('/indices')[:]

        table_doc = tables.open_file(data_dir + f_docs)
        self.docs = table_doc.get_node('/phrases')[:].astype(np.int32)
        self.idx_docs = table_doc.get_node('/indices')[:]

        assert self.idx_trans.shape[0] == self.idx_docs.shape[0]

        self.data_len = self.idx_docs.shape[0]
        print("{} entries".format(self.data_len))

    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            seq = np.append(seq, [PAD_ID] * (maxlen - len(seq)))
        seq = seq[:maxlen].astype(np.int32)

        return seq

    def __getitem__(self, item):

        len, pos = self.idx_trans[item][0], self.idx_trans[item][1]
        tran_len = min(int(len), self.max_tran_len)
        tran = self.trans[pos: pos + tran_len]

        tran, tran_block_len = pre_block(tran, self.max_tran_seq_len, self.max_tran_block_len)

        len, pos = self.idx_docs[item][0], self.idx_docs[item][1]
        good_doc_len = min(int(len), self.max_doc_len)
        good_doc = self.docs[pos: pos + good_doc_len]
        good_doc = self.pad_seq(good_doc, self.max_doc_len)

        if self.is_train:
            rand_item = random.randint(0, self.data_len - 1)
            while (rand_item == item):
                rand_item = random.randint(0, self.data_len - 1)

            len, pos = self.idx_docs[rand_item][0], self.idx_docs[rand_item][1]
            bad_doc_len = min(int(len), self.max_doc_len)
            bad_doc = self.docs[pos: pos + bad_doc_len]
            bad_doc = self.pad_seq(bad_doc, self.max_doc_len)

            return tran, tran_len, tran_block_len, good_doc, good_doc_len, bad_doc, bad_doc_len
        else:
            return tran, tran_len, tran_block_len, good_doc, good_doc_len

    def __len__(self):
        return self.data_len
