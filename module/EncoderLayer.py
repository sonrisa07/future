from torch import nn

from model.layers import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ffn, n_head, dropout):
        super(EncoderLayer, self).__init__()
        self.sublayer1 = MultiHeadAttention(d_model, n_head)
        self.sublayer2 = PositionWiseFeedForward(d_model, d_ffn, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.sublayer1(x, x, x, mask)
        x = self.norm(x)

        x = x + self.sublayer2(x)
        x = self.norm(x)

        return x
