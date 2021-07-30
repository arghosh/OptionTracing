import numpy as np
import torch
from torch.utils import data
import time
import torch
from utils import open_json, dump_json

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data):
        'Initialization'
        #QuestionId,UserId,AnswerId,IsCorrect,CorrectAnswer,AnswerValue
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.data[index]


class my_collate(object):
    def __init__(self):
        pass

    def __call__(self, batch_raw):
        #{'subject_ids': subject_ids, 'q_ids': q_ids, 'correct_ans': correct_ans,'ans': ans, 'labels': labels}
        batch = batch_raw
        L = [len(d['q_ids']) for d in batch]
        T, B = max(L), len(L)
        LSub = []
        max_sub_len = 0
        for d in batch:
            sub_len = [len(ds) for ds in d['subject_ids']]
            LSub.append(sub_len)
            max_sub_len = max(max_sub_len, max(sub_len))
        q_ids = torch.zeros(T, B).long()
        correct_ans = torch.zeros(T, B).long()+1
        ans = torch.zeros(T, B).long()+1
        labels = torch.zeros(T, B).float()
        subject_ids = torch.zeros(T, B, max_sub_len).long()
        subject_ids_mask = torch.zeros(T, B, max_sub_len).long()
        mask = torch.zeros(T, B).long()
        for idx in range(B):
            q_ids[:L[idx], idx] = torch.LongTensor(batch[idx]['q_ids'])
            correct_ans[:L[idx], idx] = torch.LongTensor(
                batch[idx]['correct_ans'])
            ans[:L[idx], idx] = torch.LongTensor(batch[idx]['ans'])
            labels[:L[idx], idx] = torch.FloatTensor(batch[idx]['labels'])
            mask[:L[idx], idx] = 1
            for l_idx in range(L[idx]):
                subject_ids[l_idx, idx, :LSub[idx]
                            [l_idx]] = torch.LongTensor(batch[idx]['subject_ids'][l_idx])
                subject_ids_mask[l_idx, idx, :LSub[idx]
                                 [l_idx]] = 1

        out = {'subject_ids': subject_ids, 'q_ids': q_ids, 'correct_ans': correct_ans, 'ans': ans, 'labels': labels, 'mask': mask, 'subject_mask': subject_ids_mask, 'L': L}
        return out


if __name__ == "__main__":
    pass