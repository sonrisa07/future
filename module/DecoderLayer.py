import torch.nn as nn

from model.layers import MultiHeadAttention, PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_head, dropout, k=None):
        super(DecoderLayer, self).__init__()
        self.sublayer1 = MultiHeadAttention(d_model, n_head)
        self.sublayer2 = PositionWiseFeedForward(d_model, d_ffn, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, usr_mat, srv_mat, mask):
        # Multi-head Attention + Residual + Norm
        x = srv_mat + self.dropout(self.sublayer1(srv_mat, usr_mat, usr_mat, mask))
        x = self.norm1(x)

        # Position-wise FFN + Residual + Norm
        x = x + self.dropout(self.sublayer2(x))
        x = self.norm2(x)

        return x

