import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import time
from datetime import datetime
import numpy as np
import argparse

random.seed(42)
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import torch

import models, configs, data_loader
from modules import get_cosine_schedule_with_warmup
from utils import similarity, normalize
from data_loader import *


def train(args):
    fh = logging.FileHandler(f"./output/{args.model}/{args.dataset}/logs.txt")
    logger.addHandler(fh)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    def save_model(model, epoch):
        torch.save(model.state_dict(), f'./output/{args.model}/{args.dataset}/models/epo{epoch}.h5')

    def load_model(model, epoch, to_device):
        assert os.path.exists(
            f'./output/{args.model}/{args.dataset}/models/epo{epoch}.h5'), f'Weights at epoch {epoch} not found'
        model.load_state_dict(
            torch.load(f'./output/{args.model}/{args.dataset}/models/epo{epoch}.h5', map_location=to_device))

    config = getattr(configs, 'config_' + args.model)()
    print(config)

    data_path = args.data_path + args.dataset + '/'
    train_set = eval(config['dataset_name'])(config, data_path,
                                             config['train_tokens'], config['tokens_len'],
                                             config['train_ast'], config['vocab_ast'],
                                             config['train_cfg'], config['n_node'],
                                             config['train_desc'], config['desc_len'])

    data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'],
                                              collate_fn=batcher(device),shuffle=True, drop_last=True, num_workers=0)

    val_set = eval(config['dataset_name'])(config, data_path,
                                           config['val_tokens'], config['tokens_len'],
                                           config['val_ast'], config['vocab_ast'],
                                           config['val_cfg'], config['n_node'],
                                           config['val_desc'], config['desc_len'])

    val_data_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=16,
                                                  collate_fn=batcher(device),shuffle=False, drop_last=False, num_workers=0)

    logger.info('Constructing Model..')
    model = getattr(models, args.model)(config)
    if args.reload_from > 0:
        load_model(model, args.reload_from, device)
    logger.info('done')
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=config['adam_epsilon'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config['warmup_steps'],
        num_training_steps=len(data_loader) * config[
            'nb_epoch'])
    if config['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=config['fp16_opt_level'])

    print('---model parameters---')
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params / 1e6)

    n_iters = len(data_loader)
    itr_global = args.reload_from + 1
    for epoch in range(int(args.reload_from) + 1, config['nb_epoch'] + 1):
        itr_start_time = time.time()
        losses = []
        for batch in data_loader:

            model.train()
            loss = model(*batch)

            if config['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            losses.append(loss.item())

            if itr_global % args.log_every == 0:
                elapsed = time.time() - itr_start_time
                logger.info('epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f' %
                            (epoch, config['nb_epoch'], itr_global % n_iters, n_iters, elapsed, np.mean(losses)))
                losses = []
                itr_start_time = time.time()
            itr_global = itr_global + 1

        if epoch % args.save_every == 0:
            save_model(model, epoch)

        if epoch % args.val_every == 0:
            model.eval()

            code_reprs, doc_reprs = [], []
            n_processed = 0
            for batch_val in val_data_loader:
                code_batch = [tensor for tensor in batch_val[:7]]
                doc_batch = [tensor for tensor in batch_val[7:9]]

                with torch.no_grad():
                    code_repr = model.code_encoding(*code_batch).data.cpu().numpy().astype(np.float32)
                    doc_repr = model.desc_encoding(*doc_batch).data.cpu().numpy().astype(np.float32)

                    code_repr = normalize(code_repr)
                    doc_repr = normalize(doc_repr)

                code_reprs.append(code_repr)
                doc_reprs.append(doc_repr)
                n_processed += batch_val[0].size(0)

            code_reprs, doc_reprs = np.vstack(code_reprs), np.vstack(doc_reprs)

            sum_1, sum_5, sum_10, sum_mrr = [], [], [], []

            for i in tqdm(range(0, n_processed)):
                doc_vec = np.expand_dims(doc_reprs[i], axis=0)
                sims = np.dot(code_reprs, doc_vec.T)[:, 0]
                negsims = np.negative(sims)
                predict = np.argsort(negsims)

                predict_1 = [int(predict[0])]
                predict_5 = [int(k) for k in predict[0:5]]
                predict_10 = [int(k) for k in predict[0:10]]

                sum_1.append(1.0) if i in predict_1 else sum_1.append(0.0)
                sum_5.append(1.0) if i in predict_5 else sum_5.append(0.0)
                sum_10.append(1.0) if i in predict_10 else sum_10.append(0.0)

                predict_list = predict.tolist()
                rank = predict_list.index(i)
                sum_mrr.append(1 / float(rank + 1))
            logger.info(
                f'epo:{epoch}, R@1={np.mean(sum_1)}, R@5={np.mean(sum_5)}, R@10={np.mean(sum_10)}, MRR={np.mean(sum_mrr)}'
            )
        model.train()


def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The Code Search (Embedding) Model")
    parser.add_argument('--data_path', type=str, default='../../src/dataset/', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='MultiEmbeder', help='model name')
    parser.add_argument('--dataset', type=str, default='MMAN', help='name of dataset.java, python')
    parser.add_argument('--reload_from', type=int, default=-1, help='epoch to reload from')

    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-v', "--visual", action="store_true", default=False,
                        help="Visualize training status in tensorboard")
    parser.add_argument('--log_every', type=int, default=50, help='interval to log autoencoder training results')
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=888, help='random seed')

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()

    os.makedirs(f'./output/{args.model}/{args.dataset}/models', exist_ok=True)
    os.makedirs(f'./output/{args.model}/{args.dataset}/tmp_results', exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train(args)
