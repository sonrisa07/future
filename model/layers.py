import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEmbedding(nn.Module):

    def __init__(self, feature_nums, emb_dim):
        super(AutoEmbedding, self).__init__()
        self.nets = nn.ModuleList()
        for num in feature_nums:
            self.nets.append(nn.Embedding(num, emb_dim))

    def forward(self, x):
        out = []
        for i, net in enumerate(self.nets):
            out.append(net(x[..., i]))
        out = torch.concat(out, dim=-1)
        return out


class FourierTemporalAttention(nn.Module):

    def __init__(self, d_model):
        super(FourierTemporalAttention, self).__init__()
        self.d_model = d_model
        self.WF_q = nn.Linear(d_model, d_model)
        self.WF_k = nn.Linear(d_model, d_model)
        self.WF_v = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        Q = self.WF_q(Q)
        K = self.WF_k(K)
        V = self.WF_v(V)

        Q_f = torch.fft.fft(Q, dim=-2)
        K_f = torch.fft.fft(K, dim=-2)
        V_f = torch.fft.fft(V, dim=-2)

        scores = (Q_f @ K_f.transpose(-2, -1)) / np.sqrt(self.d_model)
        attention = torch.softmax(scores.real, dim=-1)
        attention = attention.to(torch.complex64)
        output_f = attention @ V_f

        output = torch.fft.ifft(output_f, dim=-2).real

        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head

        assert d_model % n_head == 0

        self.depth = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.depth)
        if mask is not None:
            mask = mask.unsqueeze(-3).expand(-1, -1, scores.shape[-3], -1, -1)
            scores = scores.masked_fill(mask == 0, -1e10)
        scores = F.softmax(scores, dim=-1)
        out = scores @ v

        out = out.transpose(-2, -3).contiguous().view(q.shape[0], q.shape[1], -1, self.d_model)

        return self.dense(out)

    def split(self, x):
        x = x.view(x.shape[0], x.shape[1], -1, self.n_head, self.depth)
        return x.transpose(-2, -3)


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ffn, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x):
        return self.layer(x)
