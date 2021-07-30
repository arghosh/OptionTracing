import numpy as np
import torch
from torch.utils import data
import time
import torch
import math
from utils import open_json, dump_json


class LSTMDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, is_dash):
        'Initialization'
        self.data = data
        self.is_dash = is_dash

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        ds = self.data[index]
        if self.is_dash:
            N = len(ds['q_ids'])
            ds['dash_features'] = []
            for idx in range(N):
                wrong_before = math.log(
                    1+len([d for d in range(idx) if ds['test_mask'][d] and ds['valid_mask'][d] and ds['local_test_mask'][d] and ds['labels'][d]==0  ]))
                correct_before = math.log(
                    1+len([d for d in range(idx) if ds['test_mask'][d] and ds['valid_mask'][d] and ds['local_test_mask'][d] and ds['labels'][d] == 1]))
                wrong_same_before = math.log(
                    1+len([d for d in range(idx) if ds['test_mask'][d] and ds['valid_mask'][d] and ds['local_test_mask'][d] and ds['labels'][d] == 0 and len(
                        set.intersection(set(ds['subject_ids'][idx]), set(ds['subject_ids'][d])))]))
                correct_same_before = math.log(
                    1+len([d for d in range(idx) if ds['test_mask'][d] and ds['valid_mask'][d] and ds['local_test_mask'][d] and ds['labels'][d] == 1 and len(
                        set.intersection(set(ds['subject_ids'][idx]), set(ds['subject_ids'][d])))]))
                
                wrong_after = math.log(
                    1+len([d for d in range(idx+1,N) if ds['test_mask'][d] and ds['valid_mask'][d] and ds['local_test_mask'][d] and ds['labels'][d] == 0]))
                correct_after = math.log(
                    1+len([d for d in range(idx+1,N) if ds['test_mask'][d] and ds['valid_mask'][d] and ds['local_test_mask'][d] and ds['labels'][d] == 1]))
                wrong_same_after = math.log(
                    1+len([d for d in range(idx+1, N) if ds['test_mask'][d] and ds['valid_mask'][d] and ds['local_test_mask'][d] and ds['labels'][d] == 0 and len(
                        set.intersection(set(ds['subject_ids'][idx]), set(ds['subject_ids'][d])))]))
                correct_same_after = math.log(
                    1+len([d for d in range(idx+1, N) if ds['test_mask'][d] and ds['valid_mask'][d] and ds['local_test_mask'][d] and ds['labels'][d] == 1 and len(
                        set.intersection(set(ds['subject_ids'][idx]), set(ds['subject_ids'][d])))]))
                ds['dash_features'].append(
                    [wrong_before, correct_before, wrong_same_before, correct_same_before, wrong_after, correct_after, wrong_same_after, correct_same_after])
                

        return ds


class lstm_collate(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        #{'user_id': int(user_id), 'user_feature': user_feature, 'subject_ids': subject_ids, 'q_ids': q_ids, 'a_ids': a_ids, 'correct_ans': correct_ans,'ans': ans, 'labels': labels, 'test_mask': test_mask, 'times': times, 'confidences': confidences, 'group_ids': group_ids, 'quiz_ids': quiz_ids}
        L = [len(d['q_ids']) for d in batch]
        T, B = max(L), len(L)
        LSub = []
        max_sub_len = 0
        for d in batch:
            sub_len = [len(ds) for ds in d['subject_ids']]
            LSub.append(sub_len)
            max_sub_len = max(max_sub_len, max(sub_len))
        q_ids = torch.zeros(T, B).long()
        a_ids = torch.zeros(T, B).long()
        correct_ans = torch.zeros(T, B).long()+1
        ans = torch.zeros(T, B).long()+1
        labels = torch.zeros(T, B).float()
        test_masks = torch.zeros(T, B).long()
        valid_masks = torch.ones(T, B).long()
        local_test_masks = torch.ones(T, B).long()
        #times = torch.zeros(T, B).float()
        #confidences = torch.zeros(T, B).float()
        #group_ids = torch.zeros(T, B).long()
        #quiz_ids = torch.zeros(T, B).long()
        subject_ids = torch.zeros(T, B, max_sub_len).long()
        subject_ids_mask = torch.zeros(T, B, max_sub_len).long()
        #u_features = torch.cat([torch.FloatTensor(d['user_feature']).unsqueeze(0) for d in batch], dim=0)
        mask = torch.zeros(T, B).long()
        user_ids = [d['user_id'] for d in batch]
        if 'dash_features' in batch[0]:
            dash_features = torch.zeros(T, B, 8)
            is_dash = True
        else:
            is_dash = False

        for idx in range(B):
            if is_dash:
                dash_features[:L[idx], idx, :] = torch.FloatTensor(
                    batch[idx]['dash_features'])
            q_ids[:L[idx], idx] = torch.LongTensor(batch[idx]['q_ids'])
            # if 'a_ids' in batch[idx]:
            #     a_ids[:L[idx], idx] = torch.LongTensor(batch[idx]['a_ids'])
            correct_ans[:L[idx], idx] = torch.LongTensor(
                batch[idx]['correct_ans'])
            ans[:L[idx], idx] = torch.LongTensor(batch[idx]['ans'])
            labels[:L[idx], idx] = torch.FloatTensor(batch[idx]['labels'])
            # 1 means train, 0 means padded or unobserved
            test_masks[:L[idx], idx] = torch.LongTensor(
                batch[idx]['test_mask'])
            # 1 means train or padded, 0 means local test
            local_test_masks[:L[idx], idx] = torch.LongTensor(
                batch[idx]['local_test_mask'])
            # 1 means train or padded, 0 means local valid
            valid_masks[:L[idx], idx] = torch.LongTensor(
                batch[idx]['valid_mask'])
            #times[:L[idx], idx] = torch.FloatTensor(batch[idx]['times'])
            #confidences[:L[idx], idx] = torch.FloatTensor(batch[idx]['confidences'])
            #if 'group_ids' in batch[idx]:
            #    group_ids[:L[idx], idx] = torch.LongTensor(batch[idx]['group_ids'])
            #if 'quiz_ids' in batch[idx]:
            #    quiz_ids[:L[idx], idx] = torch.LongTensor(batch[idx]['quiz_ids'])
            mask[:L[idx], idx] = 1
            for l_idx in range(L[idx]):
                subject_ids[l_idx, idx, :LSub[idx]
                            [l_idx]] = torch.LongTensor(batch[idx]['subject_ids'][l_idx])
                subject_ids_mask[l_idx, idx, :LSub[idx]
                                 [l_idx]] = 1
        out = {'subject_ids': subject_ids, 'q_ids': q_ids, 'correct_ans': correct_ans, 'ans': ans, 'labels': labels, 'test_mask': test_masks, 'local_test_mask': local_test_masks,
                'valid_mask': valid_masks, 'mask': mask, 'subject_mask': subject_ids_mask, 'L': L, 'user_ids': user_ids}

        if is_dash:
            out['dash_features'] = dash_features
        return out


if __name__ == "__main__":
    pass
