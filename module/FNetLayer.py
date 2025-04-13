from torch import nn

from model.layers import FourierTemporalAttention, PositionWiseFeedForward


class FNetLayer(nn.Module):

    def __init__(self, d_model, dropout):
        super(FNetLayer, self).__init__()
        self.sublayer1 = FourierTemporalAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.sublayer2 = PositionWiseFeedForward(d_model, d_model * 2, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sublayer1(x, x, x)
        x = self.norm1(x)
        x = x + self.sublayer2(x)
        x = self.norm2(x)
        return x
