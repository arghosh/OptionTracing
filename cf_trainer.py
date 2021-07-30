import numpy as np
import torch
import gzip
import json
import time
import pandas as pd
import random
import yaml
from utils import open_json, dump_json, compute_auc, compute_accuracy,compute_kappa_f1_score
from cf_dataset import LSTMDataset, lstm_collate
from cf_model import LSTMModel, NCFModel, GCNModel
from copy import deepcopy
from pathlib import Path
import argparse
import os
from shutil import copyfile
from configuration import create_parser, initialize_seeds
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_val_acc, best_val_auc, best_epoch = None, -1, -1
best_test_accuracy, best_test_auc = -1, -1
best_avg_f1, best_wt_f1, best_avg_kappa, best_wt_kappa = -1, -1, -1, -1
def train_model():
    global best_val_acc, best_val_auc, best_epoch, best_test_accuracy, best_test_auc
    global best_avg_f1, best_wt_f1, best_avg_kappa, best_wt_kappa 
    batch_idx = 0
    model.train()
    train_loss, all_preds, all_targets = 0., [], []
    val_preds, val_targets = [], []
    test_preds, test_targets = [], []
    test_qids, test_dist = [], []

    for batch in train_loader:
        optimizer.zero_grad()
        loss, output,ans_dist = model(batch)
        #
        valid_mask = batch['valid_mask'].numpy()
        local_test_mask = batch['local_test_mask'].numpy()
        test_mask = batch['test_mask'].numpy()
        validation_flag = (1-valid_mask)
        local_test_flag = (1-local_test_mask)
        training_flag = test_mask*valid_mask * local_test_mask
        if params.task == '1':
            target = batch['labels'].numpy()    
        elif params.task == '2':
            target = batch['ans'].numpy()
        loss.backward()
        optimizer.step()
        #
        all_preds.append(output[training_flag == 1])
        all_targets.append(target[training_flag == 1])
        val_preds.append(output[validation_flag == 1])
        val_targets.append(target[validation_flag == 1])
        #
        if params.task == '2':
            q_ids = batch['q_ids']
            test_qids.append(q_ids[local_test_flag == 1])
            test_dist.append(ans_dist[local_test_flag==1])
        test_preds.append(output[local_test_flag == 1])
        test_targets.append(target[local_test_flag == 1])
        train_loss += float(loss.detach().cpu().numpy())
        batch_idx += 1
        

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    val_pred = np.concatenate(val_preds, axis=0)
    val_target = np.concatenate(val_targets, axis=0)
    test_pred = np.concatenate(test_preds, axis=0)
    test_target = np.concatenate(test_targets, axis=0)
    if params.task == '1':
        train_auc = compute_auc(all_target, all_pred)
        val_auc = compute_auc(val_target, val_pred)
        test_auc = compute_auc(test_target, test_pred)
        train_accuracy = compute_accuracy(all_target, all_pred)
        val_accuracy = compute_accuracy(val_target, val_pred)
        test_accuracy = compute_accuracy(test_target, test_pred)
    if params.task == '2':
        train_accuracy = np.mean(all_target == all_pred)
        val_accuracy = np.mean(val_target == val_pred)
        #
        test_dist = np.concatenate(test_dist,axis=0)
        test_qids =  np.concatenate(test_qids,axis=0)
        #
        test_accuracy = np.mean(test_target == test_pred)
        test_auc =val_auc= train_auc=-1
        avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score = compute_kappa_f1_score(test_qids, test_target, test_pred)
    if best_val_acc is None or val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_val_auc = val_auc
        best_epoch = epoch
        best_test_accuracy = test_accuracy
        best_test_auc = test_auc
        if params.task == '2':
            best_avg_f1, best_wt_f1, best_avg_kappa, best_wt_kappa  = avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score
        #save_model(model,params)
    ##
    print('Train Epoch {} Loss: {} train auc: {} train acc: {} val auc: {} val accuracy: {} n_validation : {}'.format(
        epoch, train_loss/batch_idx, train_auc, train_accuracy, val_auc, val_accuracy, val_target.shape))
    print('Train Epoch {} test accuracy: {}  test auc: {} best epoch: {}'.format(
        epoch, test_accuracy, test_auc, best_epoch))
    if params.neptune:
        neptune.log_metric('Valid Accuracy', val_accuracy)
        neptune.log_metric('Test Accuracy', test_accuracy)
        neptune.log_metric('Best Test Accuracy', best_test_accuracy)
        neptune.log_metric('Best Test Auc', best_test_auc)
        neptune.log_metric('Best Valid Accuracy', best_val_acc)
        neptune.log_metric('Best Valid Auc', best_val_auc)
        neptune.log_metric('Best Epoch', best_epoch)
        neptune.log_metric('Epoch', epoch)
        if params.task=='2':
            neptune.log_metric('best_avg_f1', best_avg_f1)
            neptune.log_metric('best_wt_f1', best_wt_f1)
            neptune.log_metric('best_avg_kappa', best_avg_kappa)
            neptune.log_metric('best_wt_kappa', best_wt_kappa)




if __name__ == "__main__":
    params = create_parser()
    params.root = 'logs/'+params.name+'/'
    if not os.path.exists(params.root):
        try:
            os.makedirs(params.root)
        except FileExistsError:
            pass
    with open(params.root+'config.yml', 'w') as outfile:
        yaml.dump(vars(params), outfile, default_flow_style=False)
    
    initialize_seeds(params.seed)
    if params.neptune:
        import neptune
        project = "arighosh/option"
        neptune.init(project_qualified_name=project,
                     api_token=os.environ["NEPTUNE_API_TOKEN"])
        neptune_exp = neptune.create_experiment(
            name=params.name, send_hardware_metrics=params.hardware, params=vars(params))
    if params.cuda:
        assert device.type == 'cuda', 'no gpu found!'

    fold_map, fold = {
        1: {'val': [1, 2], 'test': [2, 3]},
        2: {'val': [2, 3], 'test': [3, 4]},
        3: {'val': [3, 4], 'test': [4, 5]},
        4: {'val': [4, 5], 'test': [0, 1]},
        5: {'val': [0, 1], 'test': [1, 2]}
    }, params.fold

    train_data = open_json('data/cf_'+params.dataset + '.json')
    train_data = [d for d in train_data if len(
        d['q_ids']) > 0]  # 783757, 88898087 for ednet
    for d in train_data:
        temp = [idx for idx,ds in enumerate(d['test_mask']) if ds]
        random.shuffle(temp)
        N = len(temp)//5
        valid, test = set(temp[N*fold_map[fold]['val']
                           [0]: N*fold_map[fold]['val'][1]]), set(temp[N*fold_map[fold]['test']
                                                                  [0]: N*fold_map[fold]['test'][1]])
        d['valid_mask'] = [0 if idx in valid else 1 for idx in range(len(d['test_mask'])) ]
        d['local_test_mask'] = [0 if idx in test else 1 for idx in range(len(d['test_mask'])) ]

    train_dataset = LSTMDataset(train_data, is_dash='dash' in params.model)
    collate_fn = lstm_collate()
    num_workers = 2 #if params.dataset =='coda' else 2

    bs = params.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=bs, num_workers=num_workers, shuffle=True, drop_last=False)

    words = set(params.model.split('-'))
    if 'lstm' in words:
        model = LSTMModel(n_question=params.n_question, n_user=params.n_user, n_subject=params.n_subject, task=params.task,
                          n_quiz=params.n_quiz, n_group=params.n_group, is_dash='dash' in params.model, hidden_dim=params.hidden_dim, q_dim=params.question_dim, dropout=params.dropout, bidirectional='bi' in params.model).to(device)

    if 'ncf' in words:
        model = NCFModel(n_question=params.n_question, n_user=params.n_user, n_subject=params.n_subject, task=params.task, u_dim=params.hidden_dim,
                         n_quiz=params.n_quiz, n_group=params.n_group, is_dash='dash' in params.model, q_dim=params.question_dim, dropout=params.dropout).to(device)
    if 'gcn' in words:
        data = open_json('data/'+params.dataset+'_subject_to_question_map.json')
        mapper = {}

        mapper['sub'] = data['sub']
        mapper['q_map'] = [
            data['qs'][str(k)] + [0]*(params.max_len-len(data['qs'][str(k)])) for k in range(params.n_question)]
        mapper['q_mask'] = [
            [1]*len(data['qs'][str(k)]) + [0]*(params.max_len-len(data['qs'][str(k)])) for k in range(params.n_question)]
        model = GCNModel(n_question=params.n_question, n_user=params.n_user, n_subject=params.n_subject, task=params.task, mapper=mapper,
                        n_quiz=params.n_quiz, n_group=params.n_group, is_dash='dash' in params.model, hidden_dim=params.hidden_dim, q_dim=params.question_dim, dropout=params.dropout, bidirectional='bi' in params.model).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-6)
    start_time = time.time()
    for epoch in range(params.epoch):
        if (epoch-best_epoch)>params.wait:
           break
        train_model()

