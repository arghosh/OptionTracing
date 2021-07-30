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
    def __init__(self, n_question, n_subject, hidden_dim, q_dim,  task, dropout=0.25, s_dim=256):
        super().__init__()
        self.q_embeddings = nn.Embedding(n_question, q_dim)
        self.s_embeddings = nn.Embedding(n_subject, s_dim)
        self.answer_embeddings = nn.Embedding(4, s_dim)
        self.label_embeddings = nn.Embedding(2, s_dim)
        self.dropout = nn.Dropout(dropout)
        # [questions, subjects,  ans_embed, correct_ans_embed, label_embed], dim=-1))
        self.in_feature = s_dim * 4 + q_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_size=self.in_feature, hidden_size=hidden_dim,
                          num_layers=1, batch_first=False, bidirectional=False)
        self.task = task
        #pred_input = [subjects, questions, forward_ht, correct_ans]
        self.pred_in_feature = hidden_dim + 2*s_dim + q_dim
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
        mask = batch['mask'].to(device).unsqueeze(2)
        subjects = torch.sum(
            self.s_embeddings(batch['subject_ids'].to(device)) * batch['subject_mask'].to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        questions = (self.q_embeddings(
            batch['q_ids'].to(device)))  # T, B,q_dim

        ans_ = ans  # + qs*4
        correct_ans_ = correct_ans  # + qs*4
        labels_ = labels.long()  # + qs*2
        ans_embed = (self.answer_embeddings(ans_))*mask
        correct_ans_embed = (self.answer_embeddings(correct_ans_))
        label_embed = self.label_embeddings(labels_)*mask
        lstm_input = [ans_embed, correct_ans_embed,
                      label_embed, questions, subjects]
        lstm_input = self.dropout(torch.cat(lstm_input, dim=-1))

        packed_data = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths=(data_length), batch_first=False, enforce_sorted=False)
        packed_output, ht = self.rnn(packed_data)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=False)
        output = output.view(seq_len, batch_size, self.hidden_dim)
        init_state = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        forward_ht = torch.cat([init_state, output[:-1, :, :]], dim=0)

        pred_input = [questions, subjects, forward_ht, correct_ans_embed]
        pred_input = self.dropout(torch.cat(pred_input, dim=-1))

        output = self.output_layer(self.layers(pred_input)+pred_input)

        #
        if self.task == '1':
            loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            output = output.squeeze(2)
            loss = loss_fn(output, labels)
            loss = loss * mask.squeeze(2)
            loss = loss.mean() / torch.mean((mask).float())
            m = nn.Sigmoid()
            return loss, m(output).detach().cpu().numpy(), None
        elif self.task == '2':
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(output.view(-1, 4), ans.view(-1))
            loss = loss.view(-1, batch_size) * mask.squeeze(2)
            loss = loss.mean() / torch.mean((mask).float())
            pred = torch.max(output, dim=-1)[1]+1
            return loss, pred.detach().cpu().numpy(),output.detach().cpu().numpy()


class AttentionModel(nn.Module):
    def __init__(self, n_question, n_subject, hidden_dim, q_dim,  task, dropout=0.25, n_heads=1,  s_dim=256, akt=False):
        super().__init__()
        self.q_embeddings = nn.Embedding(n_question, q_dim)
        self.s_embeddings = nn.Embedding(n_subject, s_dim)
        self.answer_embeddings = nn.Embedding(4, s_dim)
        self.label_embeddings = nn.Embedding(2, s_dim)
        self.dropout = nn.Dropout(dropout)
        # [questions, subjects,  ans_embed, correct_ans_embed, label_embed], dim=-1))
        self.hidden_dim = hidden_dim
        self.task = task
        #pred_input = [subjects, questions, quizs, groups, forward_ht, user_features, features]
        self.pred_in_feature = hidden_dim + s_dim*2 + q_dim
        ####
        self.layers = nn.Sequential(
            nn.Linear(self.pred_in_feature,
                      self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.pred_in_feature, self.pred_in_feature), nn.ReLU(), nn.Dropout(dropout))

        if self.task == '1':
            self.output_layer = nn.Linear(self.pred_in_feature, 1)
        elif self.task == '2':
            self.output_layer = nn.Linear(self.pred_in_feature, 4)
        ###
        d_key = q_dim+2*s_dim
        d_val = 4*s_dim+q_dim
        self.attention_model = MultiHeadAttention(
            d_key=d_key, d_val=d_val, n_heads=n_heads, dropout=dropout, d_model=hidden_dim, akt= akt)

    def forward(self, batch):
        ans, correct_ans, labels = batch['ans'].to(
            device)-1, batch['correct_ans'].to(device) - 1, batch['labels'].to(device)
        seq_len, batch_size, data_length = ans.shape[0], ans.shape[1], batch['L']
        mask = batch['mask'].to(device).unsqueeze(2)
        subjects = torch.sum(   
            self.s_embeddings(batch['subject_ids'].to(device)) * batch['subject_mask'].to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        questions = (self.q_embeddings(
            batch['q_ids'].to(device)))  # T, B,q_dim

        ans_ = ans  # + qs*4
        correct_ans_ = correct_ans  # + qs*4
        labels_ = labels.long()  # + qs*2
        ans_embed = (self.answer_embeddings(ans_))*mask
        correct_ans_embed = (self.answer_embeddings(correct_ans_))
        label_embed = self.label_embeddings(labels_)*mask

        # Add Attention
        attn_mask = mask.squeeze(2)  # T,B, 1
        query = torch.cat([questions, subjects, correct_ans_embed], dim=-1)  # T, B, dim
        value = torch.cat([label_embed, ans_embed, correct_ans_embed,
                           questions, subjects], dim=-1)  # T,B, dim
        attention_state = self.attention_model(
            query, query, value, attn_mask)  # T,B,dim
        pred_input = [attention_state, questions, subjects, correct_ans_embed]
        # End Attention

        pred_input = self.dropout(torch.cat(pred_input, dim=-1))
        output = self.output_layer(self.layers(pred_input)+pred_input)
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_key, d_val, d_model, n_heads, dropout, bias=True, akt = False):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.h = n_heads
        self.v_linear = nn.Linear(d_val, d_model, bias=bias)
        self.k_linear = nn.Linear(d_key, d_model, bias=bias)
        self.q_linear = nn.Linear(d_key, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        #
        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self._reset_parameters()
        self.akt = akt
        if self.akt:
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        else:
            self.gammas = None
            self.position_embedding = CosinePositionalEmbedding(d_key)

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, qq, kk, vv, mask):
        T, B = qq.size(0), qq.size(1)
        if not self.akt:
            position_embed = self.position_embedding(qq).expand(-1, B, -1)/math.sqrt(self.d_model)
            qq = qq+position_embed
            kk = kk+position_embed
        # perform linear operation and split into h heads
        #T,B, h,d_k
        k = self.k_linear(qq).view(T, B, self.h, -1)
        q = self.q_linear(kk).view(T, B, self.h, -1)
        v = self.v_linear(vv).view(T, B, self.h, -1)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.permute(1, 2, 0, 3)
        q = q.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        # calculate attention using function we will define next
        scores = attention(q, k, v, mask, self.gammas)  # BS,h, T, d_k

        # concatenate heads and put through final linear layer
        concat = scores.permute(2, 0, 1, 3).contiguous().view(T, B, -1)
        output = self.layer_norm1(self.dropout(
            self.out_proj(concat)))  # T,B,d_model
        #
        output_1 = self.linear2(self.dropout(
            self.activation(self.linear1(output))))
        output = output + self.layer_norm2(output_1)
        return output




def attention(q, k, v, mask, gamma=None):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(3))  # BS, h, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    scores.masked_fill_(mask.transpose(1, 0)[:, None, None, :] == 0, -1e32)
    nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=0).astype('uint8')
    nopeek_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
    if gamma is not None:
        x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
        x2 = x1.transpose(0, 1).contiguous()
        position_effect = torch.sqrt(torch.abs(x1-x2)[None, :, :].type(torch.FloatTensor)).to(device)  # 1, seqlen, seqlen
        m = nn.Softplus()
        gamma = -1. * m(gamma)
        # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        total_effect = torch.clamp(torch.clamp((position_effect*gamma).exp(), min=1e-5), max=1e5)
        scores = scores * total_effect[None, :, :, :]

    scores.masked_fill_(nopeek_mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
    scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    output = torch.matmul(scores, v)
    return output


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:x.size(0), :, :]  # ( seq, 1,  Feature)


class DKVMN(nn.Module):
    def __init__(self, n_question, n_subject, hidden_dim, q_dim, task, dropout, s_dim, memory_size=50):
        super().__init__()
        self.n_question = n_question
        self.memory_size = memory_size
        self.memory_key_state_dim = q_dim+s_dim
        self.memory_value_state_dim = hidden_dim
        self.q_embeddings = nn.Embedding(n_question, q_dim)
        self.s_embeddings = nn.Embedding(n_subject, s_dim)
        self.answer_embeddings = nn.Embedding(4, s_dim)
        self.label_embeddings = nn.Embedding(2, s_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize Memory
        # mx.sym.Variable('init_memory_key_weight')
        self.init_memory_key = nn.Parameter(
            0.01*torch.randn(self.memory_size, self.memory_key_state_dim))
        # (self.memory_size, self.memory_value_state_dim)
        self.init_memory_value = nn.Parameter(
            0.1 * torch.randn(self.memory_size, self.memory_value_state_dim))

        self.memory = MEMORY(memory_size=self.memory_size, memory_key_state_dim=self.memory_key_state_dim,
                             memory_value_state_dim=self.memory_value_state_dim, qa_embed_dim=q_dim+4*s_dim)

        self.task = task
        #pred_input = [subjects, questions, forward_ht, correct_ans]
        self.pred_in_feature = hidden_dim + 2*s_dim + q_dim
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
        mask = batch['mask'].to(device).unsqueeze(2)
        subjects = torch.sum(
            self.s_embeddings(batch['subject_ids'].to(device)) * batch['subject_mask'].to(device).unsqueeze(3), dim=2)  # T, B, s_dim
        questions = (self.q_embeddings(
            batch['q_ids'].to(device)))  # T, B,q_dim

        ans_ = ans  # + qs*4
        correct_ans_ = correct_ans  # + qs*4
        labels_ = labels.long()  # + qs*2
        ans_embed = (self.answer_embeddings(ans_))*mask
        correct_ans_embed = (self.answer_embeddings(correct_ans_))
        label_embed = self.label_embeddings(labels_)*mask

        lstm_input = [ans_embed, correct_ans_embed,
                      label_embed, questions, subjects]
        lstm_input = self.dropout(torch.cat(lstm_input, dim=-1))
        qs = torch.cat([questions, subjects], dim=-1)

        batch_size, seq_len = ans.size(1), ans.size(0)
        memory_value = self.init_memory_value[None, :, :].expand(
            batch_size, -1, -1)
        init_memory_key = self.init_memory_key

        mem = self.memory
        value_read_content_l = []
        for i in range(seq_len):
            # Attention
            q = qs[i]
            correlation_weight = mem.attention(q, init_memory_key)
            # Read Process
            # Shape (batch_size, memory_state_dim)
            read_content = mem.read(memory_value, correlation_weight)
            # save intermedium data
            value_read_content_l.append(read_content[None, :, :])
            # Write Process
            qa = lstm_input[i]
            memory_value = mem.write(qa, memory_value, correlation_weight)

        forward_ht = torch.cat(value_read_content_l, dim=0)

        pred_input = [questions, subjects, forward_ht, correct_ans_embed]
        pred_input = self.dropout(torch.cat(pred_input, dim=-1))

        output = self.output_layer(self.layers(pred_input)+pred_input)

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


class MEMORY(nn.Module):
    """
        Implementation of Dynamic Key Value Network for Memory Tracing
        ToDo:
    """

    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, qa_embed_dim):
        super().__init__()
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_key_state_dim)
        :param init_memory_value:       Shape (batch_size, memory_size, memory_value_state_dim)
        """
        self.memory_size = memory_size
        self.qa_embed_dim = qa_embed_dim
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.erase_net = nn.Sequential(
            nn.Linear(self.qa_embed_dim,
                      self.memory_value_state_dim), nn.Sigmoid()
        )
        self.add_net = nn.Sequential(
            nn.Linear(self.qa_embed_dim,
                      self.memory_value_state_dim), nn.Tanh()
        )

    def attention(self, control_input, memory):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score = torch.matmul(
            control_input, torch.t(memory))  # BS, MS
        m = nn.LogSoftmax(dim=1)
        # Shape: (batch_size, memory_size)
        log_correlation_weight = m(similarity_score)
        return log_correlation_weight.exp()

    def read(self, memory_value, read_weight):
        read_weight = torch.reshape(
            read_weight, shape=(-1, 1, self.memory_size))
        read_content = torch.matmul(read_weight, memory_value)
        read_content = torch.reshape(read_content,  # Shape (batch_size, 1, memory_state_dim)
                                     shape=(-1, self.memory_value_state_dim))
        return read_content  # (batch_size, memory_state_dim)

    def write(self, control_input, memory, write_weight):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """

        # erase_signal  Shape (batch_size, memory_state_dim)
        erase_signal = self.erase_net(control_input)
        # add_signal  Shape (batch_size, memory_state_dim)
        add_signal = self.add_net(control_input)
        # erase_mult  Shape (batch_size, memory_size, memory_state_dim)

        erase_mult = 1 - torch.matmul(torch.reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                      torch.reshape(erase_signal, shape=(-1, 1, self.memory_value_state_dim)))

        aggre_add_signal = torch.matmul(torch.reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                        torch.reshape(add_signal, shape=(-1, 1, self.memory_value_state_dim)))
        new_memory = memory * erase_mult + aggre_add_signal
        return new_memory
