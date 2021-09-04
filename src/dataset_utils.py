from collections import Counter
import pandas as pd
import numpy as np
import argparse
import json
import h5py
import sys
import os
import re

PAD_ID, UNK_ID, EOS_ID = [0, 1, 2]


def pad_seq(seq, maxlen):
    if len(seq) < maxlen:
        seq = np.append(seq, [PAD_ID] * (maxlen - len(seq)))
    seq = seq[:maxlen].astype(np.int32)

    return seq


def pad_block(block, max_seq_len, max_block_len):
    if len(block) < max_block_len:
        val = np.ones((max_block_len - len(block), max_seq_len))
        block = np.append(block, val, axis=0)
    block = block[:max_block_len, :].astype(np.int32)
    return block


def pre_block(tran, tran_seq_len, tran_block_len):
    tran_block = []

    s_idx = 0
    for idx, i in enumerate(tran):
        if i == 2:
            tran_block.append(tran[s_idx:idx + 1])
            s_idx = idx + 1

    new_tran_block = []
    block_len = np.ones(tran_block_len, dtype=np.int32)
    for i in range(min(len(tran_block), tran_block_len)):
        tran_str = tran_block[i]
        len_ = min(len(tran_str), tran_seq_len)
        tran_str = tran_str[:len_]
        tran_str = pad_seq(tran_str, tran_seq_len)
        new_tran_block.append(tran_str)

        block_len[i] = len_

    new_tran_block = np.vstack(new_tran_block)
    new_tran_block = pad_block(new_tran_block, tran_seq_len, tran_block_len)

    return new_tran_block, block_len


def pre_tran_sentence(tran):
    return tran


def pre_tran_block(tran):
    return tran


def pre_doc(doc):
    doc_str_list = []
    doc_list = doc.splitlines()
    for dl in doc_list:
        if dl.startswith('@'):
            break
        elif dl.endswith('.') or dl.endswith('?') or dl.endswith('!'):
            doc_str_list.append(dl)
            break
        else:
            doc_str_list.append(dl)
    doc_str = ' '.join(doc_str_list)
    doc_str = doc_str.replace('!', '.').replace('?', '.')
    doc_str = doc_str.split('.')[0]
    p_cn = re.compile(u'[\u4e00-\u9fa5]')
    cn = re.search(p_cn, doc_str)
    if cn:
        return np.nan
    else:
        p_html = re.compile(r'<[^>]+>|@[a-zA-Z0-9]* +')
        doc_str = p_html.sub(' ', doc_str)
        p_letter = re.compile(r'[^a-z^A-Z^0-9]')
        doc_str = p_letter.sub(' ', doc_str)
        doc_str_list = doc_str.split()

        if len(doc_str_list) < 5:
            return np.nan
        else:
            return ' '.join(doc_str_list).lower()


def pre_token(token):
    return token


def split_word(op_str):
    op_list = []
    op_words = op_str.split()
    for i in op_words:
        i = i.replace('_', ' ').replace('[', ' ').replace(']', ' ').replace(';', ' ').replace('/', ' ').replace('\\',
                                                                                                                ' ')
        name_list = re.sub(r"([A-Z])", r" \1", i).split()
        low_name_list = []
        for n in name_list:
            low_name_list.append(n.lower())
        if len(low_name_list) == 0:
            op_list.append(op_str)
        else:
            op_list.extend(low_name_list)

    return ' '.join(op_list)


def tran_sequence(tran):
    op_list = []
    for i in tran:
        op_list.append(split_word(i))
    sequence = ' <eos> '.join(op_list)
    sequence += ' <eos>'
    return sequence


def add_method_name(tran_df):
    method_name = tran_df['name']
    tran_sentence = tran_df['tran_sentence']

    name = method_name.split('_')[0]
    name_str = split_word(name)

    new_tran_str = name_str + ' <eos> ' + tran_sentence

    return new_tran_str


def tran_block(tran):
    return tran


def remove_duplication(df):
    dataset_df = df.drop_duplicates('pre_doc')
    dataset_df = dataset_df.drop_duplicates('tran_sentence')

    print('before remove duplication:', len(df))
    print('after remove duplication:', len(dataset_df))

    return dataset_df


def pre_dataset(args, pre_):
    dataset_path = args.dataset_root + args.dataset_name
    dataset_df = pd.read_pickle(dataset_path)
    for p in pre_:
        if p == 'token':
            dataset_df['pre_token'] = dataset_df['token'].apply(pre_token)
        elif p == 'tran':
            dataset_df['tran_sentence'] = dataset_df['tran'].apply(tran_sequence)
            dataset_df['pre_tran_sentence'] = dataset_df['tran_sentence'].apply(pre_tran_sentence)

            dataset_df['tran_methodName'] = dataset_df.apply(add_method_name, axis=1)

        elif p == 'doc':
            dataset_df['pre_doc'] = dataset_df['doc'].apply(pre_doc)
        else:
            print('pre_ error')
            sys.exit(0)

    dataset_df = dataset_df.dropna(axis=0, how='any')
    dataset_df = remove_duplication(dataset_df)
    dataset_df.to_pickle(args.dataset_root + args.pre_dataset_name)


def split_dataset(args):
    df_dataset_path = args.dataset_root + args.pre_dataset_name
    if not os.path.exists(df_dataset_path):
        print('dataset path does not exist')
        sys.exit(0)

    dataset_dir = args.dataset_root + args.dataset_dir
    os.makedirs(dataset_dir, exist_ok=True)

    train_path = dataset_dir + 'train.pkl'
    val_path = dataset_dir + 'val.pkl'
    test_path = dataset_dir + 'test.pkl'

    if os.path.exists(train_path):
        print('train dataset already exists')
        return

    dataset = pd.read_pickle(df_dataset_path)

    train_dataset = dataset.iloc[:len(dataset) - args.val_dataset_num - args.test_dataset_num]
    val_dataset = dataset.iloc[len(train_dataset):len(train_dataset) + args.val_dataset_num]
    test_dataset = dataset.iloc[len(train_dataset) + args.val_dataset_num:]

    train_dataset.to_pickle(train_path)
    val_dataset.to_pickle(val_path)
    test_dataset.to_pickle(test_path)


def get_sequence_dict(args, name, df_sequences, output_path, word_num):
    dataset_list = []
    for ds in df_sequences:
        dataset_word_list = ds.split()
        for i in dataset_word_list:
            dataset_list.append(i)
    vocab_info = Counter(dataset_list)
    print(f'{name} vocab:{len(vocab_info)}')

    vocab_ = [item[0] for item in vocab_info.most_common()[:word_num - 2]]
    vocab_index = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
    vocab_index.update(zip(vocab_, [item + len(vocab_index) - 1 for item in range(len(vocab_))]))

    dict_str = json.dumps(vocab_index)
    with open(output_path, 'w') as vocab_file:
        vocab_file.write(dict_str)


def get_tran_doc_dict(args, name, df_trans, df_docs, output_path, word_num):
    dataset_list = []
    assert len(df_trans) == len(df_docs)
    for do, dd in zip(df_trans, df_docs):
        tran_word_list = do.split()
        for i in tran_word_list:
            dataset_list.append(i)

        doc_word_list = dd.split()
        for j in doc_word_list:
            dataset_list.append(j)

    vocab_info = Counter(dataset_list)
    print(f'{name} vocab:{len(vocab_info)}')

    vocab_ = [item[0] for item in vocab_info.most_common()[:word_num - 2]]
    vocab_index = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
    vocab_index.update(zip(vocab_, [item + len(vocab_index) - 1 for item in range(len(vocab_))]))

    dict_str = json.dumps(vocab_index)
    with open(output_path, 'w') as vocab_file:
        vocab_file.write(dict_str)


def sequence_index(args, df_sequences, vocab_path, output_path, sent_word_len):
    phrases, indices = [], []
    vocab = json.loads(open(vocab_path, 'r').readline())
    start_idx = 0
    for sent in df_sequences:
        word_list = sent.split()
        sent_len = min(len(word_list), sent_word_len)
        indices.append((sent_len, start_idx))
        for i in range(0, sent_len):
            word = word_list[i]
            phrases.append(vocab.get(word, UNK_ID))
        start_idx += sent_len
    sequence_index_output = h5py.File(output_path, 'w')
    sequence_index_output['phrases'] = phrases
    sequence_index_output['indices'] = indices
    sequence_index_output.close()


def pro_dataset(args, pro_):
    train_path = args.dataset_root + args.dataset_dir + 'train.pkl'
    train_df = pd.read_pickle(train_path)

    pro_file = ['train.pkl', 'val.pkl', 'test.pkl']
    output_path = args.dataset_root + args.dataset_dir
    for p in pro_:
        if p == 'token':
            dict_output_path = output_path + f'token.json'
            get_sequence_dict(args, p, train_df['pre_token'], dict_output_path, args.max_token_num)
            for i in pro_file:
                input_path = output_path + i
                df = pd.read_pickle(input_path)
                index_output_path = output_path + i[:-4] + f'_token.h5'
                sequence_index(args, df['pre_token'], dict_output_path, index_output_path, args.token_len)
        elif p == 'tran':
            dict_output_path = output_path + f'tran.json'
            get_sequence_dict(args, p, train_df['pre_tran_sentence'], dict_output_path, args.max_tran_num)
            for i in pro_file:
                input_path = output_path + i
                df = pd.read_pickle(input_path)
                index_output_path = output_path + i[:-4] + f'_tran.h5'
                sequence_index(args, df['pre_tran_sentence'], dict_output_path, index_output_path,
                               args.tran_len)
        elif p == 'doc':
            dict_output_path = output_path + f'doc.json'
            get_sequence_dict(args, p, train_df['pre_doc'], dict_output_path, args.max_doc_num)
            for i in pro_file:
                input_path = output_path + i
                df = pd.read_pickle(input_path)
                index_output_path = output_path + i[:-4] + f'_doc.h5'
                sequence_index(args, df['pre_doc'], dict_output_path, index_output_path, args.doc_len)
        else:
            print('pro error')
            sys.exit(0)

    if 'tran' in pro_ and 'doc' in pro_:
        dict_output_path = output_path + f'tran_doc.json'
        get_tran_doc_dict(args, 'tran_doc', train_df['tran_methodName'], train_df['pre_doc'],
                          dict_output_path, args.max_tran_doc_num)
        for i in pro_file:
            input_path = output_path + i
            df = pd.read_pickle(input_path)

            index_output_path = output_path + i[:-4] + f'_tran_2.h5'
            sequence_index(args, df['tran_methodName'], dict_output_path, index_output_path, args.tran_len)

            index_output_path = output_path + i[:-4] + f'_doc_2.h5'
            sequence_index(args, df['pre_doc'], dict_output_path, index_output_path, args.doc_len)


def get_dataset(args):
    dataset_path = args.dataset_root + args.dataset_name
    if os.path.exists(dataset_path):
        print('dataset already exists')
    else:
        op_files = os.listdir(args.op_path)
        op_files = [i for i in op_files if i.endswith('.pkl')]
        dataset_list = []
        for o in op_files:
            df = pd.read_pickle(args.op_path + o)
            dataset_list.append(df)
        dataset_df = pd.concat(dataset_list, axis=0, ignore_index=True)
        dataset_df = dataset_df.sample(frac=1, random_state=args.random_seed)
        dataset_df.to_pickle(dataset_path)


def run(args):
    get_dataset(args)

    pre_ = ['tran', 'doc']
    pre_dataset(args, pre_)

    split_dataset(args)
    pro_ = ['tran', 'doc']
    pro_dataset(args, pro_)


def parse_args():
    parser = argparse.ArgumentParser("Utils of the ODeepCS Dataset")
    parser.add_argument('--dataset_root', type=str, default='dataset/')
    parser.add_argument('--dataset_name', type=str, default='dataset.pkl')
    parser.add_argument('--pre_dataset_name', type=str, default='dataset_after_util.pkl')
    parser.add_argument('--dataset_dir', type=str, default='block_v3/')
    parser.add_argument('--op_path', type=str, default='../get_from_dataset/dataset/op/')
    parser.add_argument('--random_seed', type=int, default=888)
    parser.add_argument('--val_dataset_num', type=int, default=1000)
    parser.add_argument('--test_dataset_num', type=int, default=1000)

    parser.add_argument('--max_token_num', type=int, default=10000)
    parser.add_argument('--max_tran_num', type=int, default=10000)
    parser.add_argument('--max_doc_num', type=int, default=10000)
    parser.add_argument('--max_tran_doc_num', type=int, default=15000)

    parser.add_argument('--token_len', type=int, default=100)
    parser.add_argument('--tran_len', type=int, default=1000)
    parser.add_argument('--tran_seq_len', type=int, default=10)
    parser.add_argument('--tran_block_len', type=int, default=100)
    parser.add_argument('--doc_len', type=int, default=50)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
