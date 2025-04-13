import torch.nn as nn

from model.layers import MultiHeadAttention, PositionWiseFeedForward


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ffn, n_head, dropout, k=None):
        super(DecoderLayer, self).__init__()
        self.sublayer1 = MultiHeadAttention(d_model, n_head)
        self.sublayer2 = PositionWiseFeedForward(d_model, d_ffn, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, usr_mat, srv_mat, mask):
        x = srv_mat + self.sublayer1(srv_mat, usr_mat, usr_mat, mask)
        x = self.norm(x)

        # x = x + self.sublayer2(x)
        # x = self.norm(x)

        return x
