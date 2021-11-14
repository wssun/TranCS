import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import argparse
import logging
from tqdm import tqdm
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import models, configs, data_loader
from utils import normalize
from data_loader import *


def test(config, model, device):
    logger.info('Test Begin...')

    model.eval()
    model.to(device)

    data_path = args.data_path + args.dataset + '/'
    test_set = eval(config['dataset_name'])(data_path,
                                           config['test_name'], config['name_len'],
                                           config['test_tokens'], config['tokens_len'],
                                           config['test_api'], config['api_len'],
                                           config['test_desc'], config['desc_len'],
                                           is_train=False)
    data_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=16,
                                              shuffle=False, drop_last=False, num_workers=1)

    opcode_reprs, desc_reprs = [], []
    n_processed = 0

    for batch in data_loader:
        opcode_batch = [tensor.long().to(device) for tensor in batch[:6]]
        desc_batch = [tensor.long().to(device) for tensor in batch[6:]]
        with torch.no_grad():
            code_repr = model.code_encoding(*opcode_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.desc_encoding(*desc_batch).data.cpu().numpy().astype(np.float32)

            code_repr = normalize(code_repr)
            desc_repr = normalize(desc_repr)
        opcode_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)
    opcode_reprs, desc_reprs = np.vstack(opcode_reprs), np.vstack(desc_reprs)

    sum_1, sum_5, sum_10, sum_mrr = [], [], [], []

    sum_1_code_idx = []
    sum_5_code_idx = []
    sum_10_code_idx = []

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

        sum_1_code_idx.append(str(i)) if i in predict_1 else sum_1_code_idx.append(str(-1))
        sum_5_code_idx.append(str(i)) if i in predict_5 else sum_5_code_idx.append(str(-1))
        sum_10_code_idx.append(str(i)) if i in predict_10 else sum_10_code_idx.append(str(-1))

        predict_list = predict.tolist()
        rank = predict_list.index(i)
        sum_mrr.append(1 / float(rank + 1))

    logger.info(f'R@1={np.mean(sum_1)}, R@5={np.mean(sum_5)}, R@10={np.mean(sum_10)}, MRR={np.mean(sum_mrr)}')


def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument('--data_path', type=str, default='../../src/dataset/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='JointEmbeder', help='model name')
    parser.add_argument('-d', '--dataset', type=str, default='DeepCS', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=200, help='epoch to reload from')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = getattr(configs, 'config_' + args.model)()
    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu")

    model = getattr(models, args.model)(config)
    ckpt = f'./output/{args.model}/DeepCS/models/epo{args.reload_from}.h5'
    model.load_state_dict(torch.load(ckpt, map_location=device))

    test(config, model, device)
