import os
import numpy as np
import argparse
import logging
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import configs, DataLoader
from utils import normalize
from DataLoader import *
import models.TranCS as Tmodels


def test(config, model, device):
    logger.info('Test Begin...')

    model.eval()
    model.to(device)

    data_path = args.data_path + args.dataset + '/'
    test_set = eval(config['dataset_name'])(config, data_path, False,
                                           config['test_tran'],
                                           config['tran_len'],
                                           config['tran_seq_len'],
                                           config['tran_block_len'],
                                           config['test_doc'], config['doc_len'])
    data_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=2,
                                              shuffle=False, drop_last=False, num_workers=1)

    opcode_reprs, desc_reprs = [], []
    n_processed = 0

    for batch in data_loader:
        opcode_batch = [tensor.long().to(device) for tensor in batch[:3]]
        desc_batch = [tensor.long().to(device) for tensor in batch[3:]]
        with torch.no_grad():

            code_repr = model.code_encoding(*opcode_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.doc_encoding(*desc_batch).data.cpu().numpy().astype(np.float32)

            code_repr = normalize(code_repr)
            desc_repr = normalize(desc_repr)
        opcode_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)
    opcode_reprs, desc_reprs = np.vstack(opcode_reprs), np.vstack(desc_reprs)

    sum_1, sum_5, sum_10, sum_mrr = [], [], [], []

    for i in tqdm(range(0, n_processed)):
        desc_vec = np.expand_dims(desc_reprs[i], axis=0)
        sims = np.dot(opcode_reprs, desc_vec.T)[:, 0]
        negsims = np.negative(sims)
        predict = np.argsort(negsims)

        predict_1, predict_5, predict_10 = [int(predict[0])], [int(k) for k in predict[0:5]], [int(k) for k in
                                                                                               predict[0:10]]
        sum_1.append(1.0) if i in predict_1 else sum_1.append(0.0)
        sum_5.append(1.0) if i in predict_5 else sum_5.append(0.0)
        sum_10.append(1.0) if i in predict_10 else sum_10.append(0.0)

        predict_list = predict.tolist()
        rank = predict_list.index(i)
        sum_mrr.append(1 / float(rank + 1))

    logger.info(f'R@1={np.mean(sum_1)}, R@5={np.mean(sum_5)}, R@10={np.mean(sum_10)}, MRR={np.mean(sum_mrr)}')

def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--data_path', type=str, default='dataset/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='TranEmbeder', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='TranCS_dataset', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=200, help='epoch to reload from')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = getattr(configs, 'config_' + args.model)()
    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu")

    model = getattr(Tmodels, args.model)(config)
    ckpt = f'./output/{args.model}/{args.dataset}/epo{args.reload_from}.h5'
    model.load_state_dict(torch.load(ckpt, map_location=device))

    test(config, model, device)
