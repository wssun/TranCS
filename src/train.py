from datetime import datetime
from tqdm import tqdm
import numpy as np
import argparse
import random
import torch
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["NUMEXPR_MAX_THREADS"] = "12"

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

import configs as configs
from utils import normalize
import models.TranCS as models
from DataLoader import *
from models.Modules import get_cosine_schedule_with_warmup


def train(args):
    log_output_dir = f"./output/{args.model}/{args.dataset}"
    os.makedirs(log_output_dir, exist_ok=True)
    fh = logging.FileHandler(f'{log_output_dir}/logs.txt')

    logger.addHandler(fh)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config = getattr(configs, 'config_' + args.model)()
    logger.info(config)

    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() else "cpu")

    def save_model(model, epoch):
        torch.save(model.state_dict(), f'./output/{args.model}/{args.dataset}/epo{epoch}.h5')

    def load_model(model, epoch, to_device):
        assert os.path.exists(
            f'./output/{args.model}/{args.dataset}/epo{epoch}.h5'), f'Weights at epoch {epoch} not found'
        model.load_state_dict(torch.load(f'./output/{args.model}/{args.dataset}/epo{epoch}.h5',
                                         map_location=to_device))

    data_path = args.data_path + args.dataset + '/'
    train_set = eval(config['dataset_name'])(config, data_path, True,
                                             config['train_tran'],
                                             config['tran_len'],
                                             config['tran_seq_len'],
                                             config['tran_block_len'],
                                             config['train_doc'], config['doc_len'])

    data_loader_train = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'],
                                                    shuffle=True, drop_last=False, num_workers=1)

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
        num_training_steps=len(data_loader_train) * config['n_epoch'])

    n_iters = len(data_loader_train)
    itr_global = args.reload_from + 1
    for epoch in range(int(args.reload_from) + 1, config['n_epoch'] + 1):
        itr_start_time = time.time()
        losses = []
        for batch in data_loader_train:
            model.train()
            batch_gpu = [tensor.long().to(device) for tensor in batch]
            loss = model(*batch_gpu)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            losses.append(loss.item())

            if itr_global % args.log_every == 0:
                elapsed = time.time() - itr_start_time
                logger.info('epo:[%d/%d] itr:[%d/%d] step_time:%ds Loss=%.5f' %
                            (epoch, config['n_epoch'], itr_global % n_iters, n_iters, elapsed, np.mean(losses)))
                losses = []
                itr_start_time = time.time()
            itr_global = itr_global + 1

        if epoch % args.save_every == 0:
            save_model(model, epoch)


def parse_args():
    parser = argparse.ArgumentParser("Train and Validate The TranCS Model")
    parser.add_argument('--data_path', type=str, default='../../src/dataset/')
    parser.add_argument('--model', type=str, default='TranEmbeder')
    parser.add_argument('--dataset', type=str, default='TranCS_dataset')
    parser.add_argument('--reload_from', type=int, default=-1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=888)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train(args)
