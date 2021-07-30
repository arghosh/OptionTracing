import json
import numpy as np
from sklearn import metrics
import torch
from sklearn.metrics import f1_score,cohen_kappa_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data


def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data

def compute_kappa_f1_score(qids, target, pred):
    unique_qids = np.unique(qids)
    f1_scores = np.zeros(len(unique_qids))
    cohen_scores =np.zeros(len(unique_qids))
    weights = np.zeros(len(unique_qids))
    for idx, qid in enumerate(unique_qids):
        qid_mask = qids==qid
        qid_target =  target[qid_mask]
        qid_pred = pred[qid_mask]
        f1_scores[idx] = f1_score(qid_target,qid_pred,average='macro')
        cohen_scores[idx] = cohen_kappa_score(qid_target, qid_pred)
        weights[idx] =  np.sum(qid_mask)/(len(target)+0.)
    cohen_scores[np.isnan(cohen_scores)] = 1.
    avg_f1_score = np.average(f1_scores)
    weighted_f1_score = np.sum(f1_scores*weights)
    avg_cohen_score =  np.average(cohen_scores)
    weighted_cohen_score =  np.sum(cohen_scores*weights)
    return avg_f1_score, weighted_f1_score, avg_cohen_score, weighted_cohen_score

def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred_new = all_pred.copy()
    all_pred_new[all_pred > 0.5] = 1.0
    all_pred_new[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred_new)

def format_data_as_kt(data, seqlen, threshold = 0):
    if len(data['q_ids'])<=threshold:
        return []
    schema = ['subject_ids', 'q_ids', 'correct_ans', 'ans', 'labels']
    if 'test_mask' in data:
        temp_data = {}
        for field in schema:
            temp_data[field] = [d for idx, d in enumerate(data[field])  if data['test_mask'][idx] ]
        data = temp_data
    #Now Split 
    output = []
    N = len(data['q_ids'])
    if N<=seqlen:
        return [data]
    n_batch = N//seqlen if N%seqlen==0 else N//seqlen+1
    for b in range(n_batch):
        temp_data = {}
        for field in schema:
            temp_data[field] =  data[field][b*seqlen: (b+1)*seqlen]
        if len(temp_data['q_ids'])>threshold:
            output.append(temp_data)
    return output


def subject_map_ednet():
    q_map = open_json('data/q_map.json')
    mapper = open_json('data/convert_mapper.json')
    inverse_q_mapper = {}
    inverse_s_mapper = {}
    for k,v in mapper['q_id'].items():
        inverse_q_mapper[v] = int(k)
    for k,v in mapper['tag_id'].items():
        inverse_s_mapper[v] = int(k)
    output = {'qs':{}, 'sub':{}}
    for k,v in q_map.items():
        new_q_id = mapper['q_id'][k]
        new_s_id = [mapper['tag_id'][str(d)] for d in v['tags']]
        output['qs'][new_q_id] =  new_s_id
        for s_id in new_s_id:
            if s_id not in output['sub']:
                output['sub'][s_id] = []
            output['sub'][s_id].append(new_q_id)

    dump_json('data/ednet_subject_to_question_map.json', output)


if __name__ == "__main__":
    subject_map_ednet()
