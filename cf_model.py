import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMModel(nn.Module):
    def __init__(self, n_question,  n_user, n_subject, n_quiz, n_group, hidden_dim, q_dim,  task, dropout=0.25, bidirectional=False, num_gru_layers=1, is_dash=False):
        super().__init__()
        self.is_dash = is_dash
        self.bidirectional = bidirectional
        self.q_embeddings = nn.Embedding(n_question, q_dim)
        self.s_embeddings = nn.Embedding(n_subject, q_dim)
        # self.feature_layer = nn.Linear(2, q_dim)
        self.answer_embeddings = nn.Embedding(4, q_dim)
        self.label_embeddings = nn.Embedding(2, q_dim)
        # self.user_feature_layer = nn.Linear(8, q_dim)
        self.dropout = nn.Dropout(dropout)
        # [questions, subjects,  ans_embed, correct_ans_embed, label_embed], dim=-1))
        self.in_feature = 5 * q_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=self.in_feature, hidden_size=hidden_dim,
                          num_layers=num_gru_layers, batch_first=False, bidirectional=bidirectional)
        self.task = task
        # pred_input = [subjects, questions, quizs, groups, forward_ht]
        self.pred_in_feature = hidden_dim + 3 * q_dim
        if self.bidirectional:
            self.pred_in_feature += hidden_dim
        if n_quiz:
            self.pred_in_feature += q_dim
            self.quiz_embeddings = nn.Embedding(n_quiz, q_dim)
        if n_group:
            self.pred_in_feature += q_dim
            self.group_embeddings = nn.Embedding(n_group, q_dim)
        if self.is_dash:
            self.dash_layer = nn.Linear(8, q_dim)
            self.pred_in_feature += q_dim

        self.layers = nn.Sequential(
            nn.Linear(self.pred_in_feature,
                      self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.pred_in_feature, self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout))

        if self.task == '1':
            self.output_layer = nn.Linear(self.pred_in_feature, 1)
        elif self.task == '2':
            self.output_layer = nn.Linear(self.pred_in_feature, 4)

    def forward(self, batch):
        ans, correct_ans, labels = batch['ans'].to(
            device)-1, batch['correct_ans'].to(device) - 1, batch['labels'].to(device)
        seq_len, batch_size, data_length = ans.shape[0], ans.shape[1], batch['L']
        # Mask
        test_mask = batch['test_mask'].to(device).unsqueeze(2)
        valid_mask = batch['valid_mask'].to(device).unsqueeze(2)
        local_test_mask = batch['local_test_mask'].to(device).unsqueeze(2)
        mask = test_mask * valid_mask * local_test_mask
        #
        # user_features = (self.user_feature_layer(
        #     batch['user_features'].to(device))).unsqueeze(0).expand(seq_len, -1, -1)  # T, B,uf_dim
        subjects = torch.sum((self.s_embeddings(
            batch['subject_ids'].to(device))) * batch['subject_mask'].to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        questions = (self.q_embeddings(
            batch['q_ids'].to(device)))  # T, B,q_dim
        # apply test_mask
        # qs = batch['q_ids'].to(device)
        ans_ = ans  # + qs*4
        correct_ans_ = correct_ans  # + qs*4
        labels_ = labels.long()  # + qs*2
        ans_embed = (self.answer_embeddings(ans_))*mask
        correct_ans_embed = (self.answer_embeddings(correct_ans_))
        label_embed = self.label_embeddings(labels_)*mask

        # input_feature = torch.cat([batch['times'].to(device).unsqueeze(
        #     2), batch['confidences'].to(device).unsqueeze(2)], dim=-1)
        # features = (self.feature_layer(input_feature))

        lstm_input = [ans_embed, correct_ans_embed,
            label_embed, questions, subjects]
        lstm_input = self.dropout(torch.cat(lstm_input, dim=-1))

        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths=(data_length), batch_first=False, enforce_sorted=False)
        packed_output, ht = self.rnn(packed_data)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=False)
        init_state = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        if self.bidirectional:
            output = output.view(seq_len, batch_size, 2, self.hidden_dim)
            forward_ht = torch.cat([init_state, output[:-1, :, 0, :]], dim=0)
            reverse_ht = torch.cat([output[1:, :, 1, :], init_state], dim=0)
            pred_input = [questions, reverse_ht,
                subjects, forward_ht,  correct_ans_embed]
        else:
            output = output.view(seq_len, batch_size, 1, self.hidden_dim)
            forward_ht = torch.cat([init_state, output[:-1, :, 0, :]], dim=0)
            pred_input = [questions, subjects, forward_ht,  correct_ans_embed]

        if self.is_dash:
            dash_features = self.dropout(
                self.dash_layer(batch['dash_features'].to(device)))
            pred_input.append(dash_features)
        # if 'quiz_ids' in batch:
        #     quizs = (self.quiz_embeddings(
        #         batch['quiz_ids'].to(device)))  # T, B,q_dim
        #     pred_input.append(quizs)
        # if 'group_ids' in batch:
        #     groups = (self.group_embeddings(
        #         batch['group_ids'].to(device)))  # T, B,q_dim
        #     pred_input.append(groups)

        pred_input = self.dropout(torch.cat(pred_input, dim=-1))
        output = self.output_layer(self.layers(pred_input)+pred_input)
        #
        if self.task == '1':
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            output = output.squeeze()
            loss = loss_fn(output, labels)
            loss = loss * mask.squeeze(2)
            loss = loss.mean() / torch.mean(mask.float())
            m = nn.Sigmoid()
            return loss, m(output).detach().cpu().numpy(), None
        elif self.task == '2':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(output.view(-1, 4), ans.view(-1))
            loss = loss.view(-1, batch_size) * mask.squeeze(2)
            loss = loss.mean() / torch.mean(mask.float())
            pred = torch.max(output, dim=-1)[1]+1
            return loss, pred.detach().cpu().numpy(), output.detach().cpu().numpy()


class NCFModel(nn.Module):
    def __init__(self, n_question, n_subject, n_user, n_quiz, n_group, task, is_dash=False, dropout=0.2, q_dim=256, u_dim=256):
        super().__init__()
        self.q_embeddings = nn.Embedding(n_question, q_dim)
        self.u_embeddings = nn.Embedding(n_user, u_dim)
        self.s_embeddings = nn.Embedding(n_subject, q_dim)
        self.dash = is_dash
        self.dropout = nn.Dropout(dropout)
        in_feature = 2*q_dim + u_dim
        if self.dash:
            self.dash_layer = nn.Linear(8, q_dim)
            in_feature += q_dim
        if n_quiz:
            self.quiz_embeddings = nn.Embedding(n_quiz, q_dim)
            in_feature += q_dim
        if n_group:
            # self.user_feature_layer = nn.Linear(8, q_dim)
            self.group_embeddings = nn.Embedding(n_group, q_dim)
            in_feature += q_dim

        self.layers = nn.Sequential(
            nn.Linear(in_feature, in_feature), nn.ReLU(
            ), nn.Dropout(dropout),
            nn.Linear(in_feature, in_feature), nn.ReLU(), nn.Dropout(dropout)
        )
        self.task = task
        if self.task == '1':
            self.output_layer = nn.Linear(in_feature, 1)
        else:
            self.output_layer = nn.Linear(in_feature, 4)

    def forward(self, batch):
        u_ids = self.dropout(self.u_embeddings(
            torch.LongTensor(batch['user_ids']).to(device)))  # B, u_dim
        ans, correct_ans, labels = batch['ans'].to(
            device)-1, batch['correct_ans'].to(device) - 1, batch['labels'].to(device)
        seq_len, batch_size, data_length = ans.shape[0], ans.shape[1], batch['L']
        # Mask
        test_mask = batch['test_mask'].to(device).unsqueeze(2)
        valid_mask = batch['valid_mask'].to(device).unsqueeze(2)
        local_test_mask = batch['local_test_mask'].to(device).unsqueeze(2)
        mask = test_mask * valid_mask * local_test_mask
        #
        # user_features = (self.user_feature_layer(
        #     batch['user_features'].to(device))).unsqueeze(0).expand(seq_len, -1, -1)  # T, B,uf_dim
        subjects = torch.sum((self.s_embeddings(
            batch['subject_ids'].to(device))) * batch['subject_mask'].to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        questions = (self.q_embeddings(
            batch['q_ids'].to(device)))  # T, B,q_dim
        users = u_ids[None, :, :].expand(seq_len, -1, -1)

        input_embeddings = [questions, subjects, users]
        if 'quiz_ids' in batch:
            quiz_ids = self.dropout(
                self.quiz_embeddings(batch['quiz_ids'].to(device)))
            input_embeddings.append(quiz_ids)
        if 'group_ids' in batch:
            group_ids = self.dropout(
                self.group_embeddings(batch['group_ids'].to(device)))
            input_embeddings.append(group_ids)

        if self.dash:
            dash_features = self.dropout(
                self.dash_layer(batch['dash_features'].to(device)))
            input_embeddings.append(dash_features)
        input_embeddings = torch.cat(input_embeddings, dim=-1)

        output = self.output_layer(self.layers(
            input_embeddings)+input_embeddings)
        #
        if self.task == '1':
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            output = output.squeeze(2)
            loss = loss_fn(output, labels)
            loss = loss * mask.squeeze(2)
            loss = loss.mean() / torch.mean((mask).float())
            m = nn.Sigmoid()
            return loss, m(output).detach().cpu().numpy(),None
        elif self.task == '2':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(output.view(-1, 4), ans.view(-1))
            loss = loss.view(-1, batch_size) * mask.squeeze(2)
            loss = loss.mean() / torch.mean((mask).float())
            pred = torch.max(output, dim=-1)[1]+1
            return loss, pred.detach().cpu().numpy(),output.detach().cpu().numpy()


class GCNModel(nn.Module):
    def __init__(self, n_question,  n_user, n_subject, n_quiz, n_group, hidden_dim, q_dim,  task, mapper, dropout=0.25, bidirectional=False, num_gru_layers=1, is_dash=False):
        super().__init__()
        self.is_dash = is_dash
        self.bidirectional = bidirectional
        self.n_subject = n_subject
        # nn.Embedding(n_question, q_dim)
        self.q_embeddings = nn.Parameter(0.1*torch.randn(n_question, q_dim))
        # nn.Embedding(n_subject, s_dim)
        self.s_embeddings = nn.Parameter(0.1*torch.randn(n_subject, q_dim))
        # self.feature_layer = nn.Linear(2, q_dim)
        self.answer_embeddings = nn.Embedding(4, q_dim)
        self.label_embeddings = nn.Embedding(2, q_dim)
        # self.user_feature_layer = nn.Linear(8, q_dim)
        self.dropout = nn.Dropout(dropout)
        # [questions, subjects,  ans_embed, correct_ans_embed, label_embed], dim=-1))
        self.in_feature = 5 * q_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=self.in_feature, hidden_size=hidden_dim,
                          num_layers=num_gru_layers, batch_first=False, bidirectional=bidirectional)
        self.task = task
        # pred_input = [subjects, questions, quizs, groups, forward_ht]
        self.pred_in_feature = hidden_dim + 3 * q_dim
        if self.bidirectional:
            self.pred_in_feature += hidden_dim
        if n_quiz:
            self.pred_in_feature += q_dim
            self.quiz_embeddings = nn.Embedding(n_quiz, q_dim)
        if n_group:
            self.pred_in_feature += q_dim
            self.group_embeddings = nn.Embedding(n_group, q_dim)
        if self.is_dash:
            self.dash_layer = nn.Linear(8, q_dim)
            self.pred_in_feature += q_dim

        self.layers = nn.Sequential(
            nn.Linear(self.pred_in_feature,
                      self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.pred_in_feature, self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout))

        if self.task == '1':
            self.output_layer = nn.Linear(self.pred_in_feature, 1)
        elif self.task == '2':
            self.output_layer = nn.Linear(self.pred_in_feature, 4)

        self.q_map = torch.LongTensor(mapper['q_map']).to(device)  # Q,8
        self.q_mask = torch.FloatTensor(mapper['q_mask']).to(device)  # Q,8
        self.sub_map = {}
        for k, v in mapper['sub'].items():
            self.sub_map[int(k)] = torch.LongTensor(v).to(device)
        self.s2s = nn.Linear(q_dim, q_dim)
        self.s2q = nn.Linear(q_dim, q_dim)
        self.q2q = nn.Linear(q_dim, q_dim)
        self.q2s = nn.Linear(q_dim, q_dim)

    def forward(self, batch):
        tanh = nn.Tanh()
        ans, correct_ans, labels = batch['ans'].to(
            device)-1, batch['correct_ans'].to(device) - 1, batch['labels'].to(device)
        seq_len, batch_size, data_length = ans.shape[0], ans.shape[1], batch['L']
        # Mask
        test_mask = batch['test_mask'].to(device).unsqueeze(2)
        valid_mask = batch['valid_mask'].to(device).unsqueeze(2)
        local_test_mask = batch['local_test_mask'].to(device).unsqueeze(2)
        mask = test_mask * valid_mask * local_test_mask
        subjects_0 = self.s_embeddings  # S, dim
        questions_0 = self.q_embeddings  # Q, dim
        s_dim, q_dim = subjects_0.shape[1], questions_0.shape[1]
        subjects_1 = []
        for sub_idx in range(self.n_subject):
            if sub_idx not in self.sub_map:
                subjects_1.append(torch.zeros(1, s_dim).to(device))
            else:
                nq = torch.mean(
                    self.q2s(questions_0[self.sub_map[sub_idx], :]), dim=0)
                sq = self.s2s(subjects_0[sub_idx])
                subjects_1.append(tanh(nq+sq).unsqueeze(0))
        subjects_1 = torch.cat(subjects_1, dim=0)
        nq = torch.sum(subjects_1[self.q_map] * self.q_mask.unsqueeze(
            2), dim=1) / torch.sum(self.q_mask, dim=-1, keepdim=True)  # n_question,8 q_dim
        nq = self.s2q(nq)  # Q,q_dim
        sq = self.q2q(questions_0)  # Q, q_dim
        questions_1 = tanh(nq+sq)
        subjects = torch.sum(subjects_1[batch['subject_ids'].to(
            device)] * batch['subject_mask'].to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        questions = questions_1[batch['q_ids'].to(device)]  # T, B,q_dim
        # subjects = torch.sum((self.s_embeddings(
        #     batch['subject_ids'].to(device))) * batch['subject_mask'].to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        # questions = (self.q_embeddings(
        #     batch['q_ids'].to(device)))  # T, B,q_dim
        ans_ = ans  # + qs*4
        correct_ans_ = correct_ans  # + qs*4
        labels_ = labels.long()  # + qs*2
        ans_embed = (self.answer_embeddings(ans_))*mask
        correct_ans_embed = (self.answer_embeddings(correct_ans_))
        label_embed = self.label_embeddings(labels_)*mask

        # input_feature = torch.cat([batch['times'].to(device).unsqueeze(
        #     2), batch['confidences'].to(device).unsqueeze(2)], dim=-1)
        # features = (self.feature_layer(input_feature))

        lstm_input = [ans_embed, correct_ans_embed,
                      label_embed, questions, subjects]
        lstm_input = self.dropout(torch.cat(lstm_input, dim=-1))

        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths=(data_length), batch_first=False, enforce_sorted=False)
        packed_output, ht = self.rnn(packed_data)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=False)
        init_state = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        if self.bidirectional:
            output = output.view(seq_len, batch_size, 2, self.hidden_dim)
            forward_ht = torch.cat([init_state, output[:-1, :, 0, :]], dim=0)
            reverse_ht = torch.cat([output[1:, :, 1, :], init_state], dim=0)
            pred_input = [questions, reverse_ht,
                          subjects, forward_ht,  correct_ans_embed]
        else:
            output = output.view(seq_len, batch_size, 1, self.hidden_dim)
            forward_ht = torch.cat([init_state, output[:-1, :, 0, :]], dim=0)
            pred_input = [questions, subjects, forward_ht,  correct_ans_embed]

        if self.is_dash:
            dash_features = self.dropout(
                self.dash_layer(batch['dash_features'].to(device)))
            pred_input.append(dash_features)
        if 'quiz_ids' in batch:
            quizs = (self.quiz_embeddings(
                batch['quiz_ids'].to(device)))  # T, B,q_dim
            pred_input.append(quizs)
        if 'group_ids' in batch:
            groups = (self.group_embeddings(
                batch['group_ids'].to(device)))  # T, B,q_dim
            pred_input.append(groups)

        pred_input = self.dropout(torch.cat(pred_input, dim=-1))
        output = self.output_layer(self.layers(pred_input)+pred_input)
        #
        if self.task == '1':
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            output = output.squeeze()
            loss = loss_fn(output, labels)
            loss = loss * mask.squeeze(2)
            loss = loss.mean() / torch.mean(mask.float())
            m = nn.Sigmoid()
            return loss, m(output).detach().cpu().numpy(),None
        elif self.task == '2':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(output.view(-1, 4), ans.view(-1))
            loss = loss.view(-1, batch_size) * mask.squeeze(2)
            loss = loss.mean() / torch.mean(mask.float())
            pred = torch.max(output, dim=-1)[1]+1
            return loss, pred.detach().cpu().numpy(),output.detach().cpu().numpy()


