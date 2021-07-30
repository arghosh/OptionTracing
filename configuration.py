import argparse
import numpy as np
import torch
import random


def initialize_seeds(seedNum):
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seedNum)
        torch.cuda.manual_seed_all(seedNum)


def create_parser():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Script to test KT')
    parser.add_argument('--name', default='demo',help='Name for the experiment')
    parser.add_argument('--nodes', default='', help='slurm nodes for the experiment')
    parser.add_argument('--slurm_partition', default='',
                        help='slurm partitions for the experiment')
    # Basic Parameters
    parser.add_argument('--task', type=str, default='2', help='type')
    parser.add_argument('--hidden_dim', type=int, default=128, help='type')
    parser.add_argument('--question_dim', type=int, default=32, help='type')
    parser.add_argument('--lr', type=float, default=1e-3, help='type')
    parser.add_argument('--dropout', type=float, default=0.25, help='type')
    parser.add_argument('--batch_size', type=int, default=64, help='type')
    parser.add_argument('--fold', type=int, default=1, help='type')
    parser.add_argument('--dataset', type=str,
                        default='coda', help='type')
    parser.add_argument('--model', type=str, default='lstm', help='type')
    parser.add_argument('--setup', type=str, default='kt', help='type')
    parser.add_argument('--head', type=int, default=8, help='type')
    parser.add_argument('--neptune', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--hardware', action='store_true')
    parser.add_argument('--seed', type=int, default=221, help='type')
    parser.add_argument('--file_name', type=str,
                        default='', help='type')
    parser.add_argument('--hash', type=str,
                        default='', help='type')

    params = parser.parse_args()
    if params.dataset=='eedi':
        params.wait = 20
        params.epoch = 400
    else:
        params.wait = 10
        params.epoch = 200

    params.n_quiz=params.n_group = 0
    #
    if 'ednet' in params.dataset:
        params.n_question = 13169
        params.n_subject = 189
        params.n_user = 784310
        params.max_len = 7
        
    if params.dataset == 'coda':
        params.n_question = 27613
        params.n_subject = 389
        params.n_user = 118971
        #n_quiz = 17305
        #n_group = 11844
        params.max_len = 6
    if params.dataset == 'eedi':
        params.n_question = 948
        params.n_subject = 389
        params.n_user = 6148
        params.max_len = 2
    return params
