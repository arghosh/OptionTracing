import numpy as np
import torch
import gzip
import json
import time
import pandas as pd
import random
import yaml
from utils import open_json, dump_json, compute_auc, compute_accuracy,compute_kappa_f1_score
from kt_dataset import Dataset, my_collate
from kt_model import LSTMModel, DKVMN, AttentionModel
from copy import deepcopy
from pathlib import Path
import argparse
from shutil import copyfile
import os
from configuration import create_parser, initialize_seeds
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_val_acc, best_val_auc, best_epoch = None, -1, -1
test_accuracy, test_auc = -1, -1
avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score = -1,-1,-1,-1

def train_model():
    global best_val_acc, best_val_auc, best_epoch, test_accuracy, test_auc
    global avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score
    batch_idx = 0
    model.train()
    train_loss, all_preds, all_targets = 0., [], []
    for batch in train_loader:
        optimizer.zero_grad()
        loss, output, _ = model(batch)
        #
        training_flag = batch['mask'].numpy()
        if params.task == '1':
            target = batch['labels'].numpy()
        elif params.task == '2':
            target = batch['ans'].numpy()
        # q_ids = batch['q_ids'].numpy()
        # u_ids = batch['user_ids']
        loss.backward()
        optimizer.step()

        all_preds.append(output[training_flag == 1])
        all_targets.append(target[training_flag == 1])
        train_loss += float(loss.detach().cpu().numpy())
        batch_idx += 1

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    model.eval()
    if params.task == '1':
        train_auc = compute_auc(all_target, all_pred)
        train_accuracy = compute_accuracy(all_target, all_pred)
        val_accuracy, val_auc = test_model('val')
    if params.task == '2':
        train_accuracy = np.mean(all_target == all_pred)
        val_accuracy, val_auc = test_model('val')
        train_auc = -1

    print('Train Epoch {} Loss: {} Train Auc: {} Train acc: {} Val Auc: {} Val acc: {}'.format(
        epoch, train_loss/batch_idx, train_auc, train_accuracy, val_auc, val_accuracy))

    if best_val_acc is None or val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_val_auc = val_auc
        best_epoch = epoch
        if params.task=='2':
            test_accuracy, test_auc, avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score,info = test_model('test')
            with gzip.open(params.root+'test_scores.gzip', 'wt', encoding="ascii") as zipfile:
                json.dump(info, zipfile)
        else:
            test_accuracy, test_auc =  test_model('test')


    print('Train Epoch {} Best Val Acc: {} Test Acc: {} Test Auc: {}'.format(
        epoch, best_val_acc, test_accuracy, test_auc))
    if params.neptune:
        neptune.log_metric('Valid Accuracy', val_accuracy)
        neptune.log_metric('Test Accuracy', test_accuracy)
        neptune.log_metric('Best Test Accuracy', test_accuracy)
        neptune.log_metric('Best Test Auc', test_auc)
        neptune.log_metric('Best Valid Accuracy', best_val_acc)
        neptune.log_metric('Best Valid Auc', best_val_auc)
        neptune.log_metric('Best Epoch', best_epoch)
        neptune.log_metric('Epoch', epoch)
        if params.task=='2':
            neptune.log_metric('best_avg_f1', avg_f1_score)
            neptune.log_metric('best_wt_f1', weighted_f1_score)
            neptune.log_metric('best_avg_kappa', avg_cohen_score)
            neptune.log_metric('best_wt_kappa', weighted_cohen_score)

    model.train()


def test_model(name ='val'):
    all_preds, all_targets = [], []
    test_qids, test_dist = [], []
    loader = val_loader if name=='val' else test_loader
    for batch in loader:
        with torch.no_grad():
            _, output,ans_dist = model(batch)
        if params.task == '1':
            target = batch['labels'].numpy()
        elif params.task == '2':
            target = batch['ans'].numpy()
        validation_flag = batch['mask'].numpy()
        all_preds.append(output[validation_flag == 1])
        all_targets.append(target[validation_flag == 1])
        if params.task == '2':
            q_ids = batch['q_ids']
            test_qids.append(q_ids[validation_flag == 1])
            test_dist.append(ans_dist[validation_flag==1])

    all_pred = np.concatenate(all_preds, axis=0)
    all_target = np.concatenate(all_targets, axis=0)
    if params.task == '1':
        auc = compute_auc(all_target, all_pred)
        accuracy = compute_accuracy(all_target, all_pred)
    else:
        auc = -1
        accuracy = np.mean(all_target == all_pred)
    if params.task=='2' and name=='test':
        test_dist = np.concatenate(test_dist,axis=0)
        test_qids =  np.concatenate(test_qids,axis=0)
        avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score = compute_kappa_f1_score(test_qids, all_target, all_pred)
        save_info = {'qids': test_qids.tolist(), 'dist': test_dist.tolist(), 'target': all_target.tolist()}
        return accuracy,auc,avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score,save_info 


    return accuracy, auc



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
    

    all_data = open_json('data/kt_'+params.dataset + '.json')
    N = len(all_data)//5
    fold_map, fold = {
        1: {'val': [1, 2], 'test': [2, 3]},
        2: {'val': [2, 3], 'test': [3, 4]},
        3: {'val': [3, 4], 'test': [4, 5]},
        4: {'val': [4, 5], 'test': [0, 1]},
        5: {'val': [0, 1], 'test': [1, 2]}
    }, params.fold
    train_data = [d for idx, d in enumerate(all_data) if not(
        idx >= N*fold_map[fold]['val'][0] and idx <= N*fold_map[fold]['val'][1]) and not(idx >= N*fold_map[fold]['test'][0] and idx <= N*fold_map[fold]['test'][1])]
    val_data = [d for idx, d in enumerate(all_data) if (
        idx >= N*fold_map[fold]['val'][0] and idx <= N*fold_map[fold]['val'][1])]
    test_data = [d for idx, d in enumerate(all_data) if (
        idx >= N*fold_map[fold]['test'][0] and idx <= N*fold_map[fold]['test'][1])]

    del all_data

    #train_data =[d for d in train_data if len(d['q_ids'])<2000 ]
    # print(len(train_data))
    train_dataset, val_dataset, test_dataset = Dataset(train_data), Dataset(val_data), Dataset(test_data)
    collate_fn = my_collate()
    num_workers = 2
    bs = params.batch_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=bs, num_workers=num_workers, shuffle=True, drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, collate_fn=collate_fn, batch_size=bs*2, num_workers=num_workers, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, collate_fn=collate_fn, batch_size=bs*2, num_workers=num_workers, shuffle=False, drop_last=False)
    if params.model =='lstm':
        model = LSTMModel(n_question=params.n_question, n_subject=params.n_subject, task=params.task, s_dim=params.question_dim,
                      hidden_dim=params.hidden_dim, q_dim=params.question_dim, dropout=params.dropout).to(device)
    elif params.model =='dkvmn':
        model = DKVMN(n_question=params.n_question, n_subject=params.n_subject, task=params.task, s_dim=params.question_dim,
                          hidden_dim=params.hidden_dim, q_dim=params.question_dim, dropout=params.dropout).to(device)
    elif 'attn' in params.model :
        model = AttentionModel(n_question=params.n_question, n_subject=params.n_subject, task=params.task, s_dim=params.question_dim, n_heads=params.head,
                      hidden_dim=params.hidden_dim, q_dim=params.question_dim, dropout=params.dropout, akt ='akt' in params.model).to(device)

    print(model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-6)
    start_time = time.time()
    for epoch in range(params.epoch):
        if (epoch-best_epoch) > params.wait:
           break
        train_model()
