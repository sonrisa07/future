from torch import nn

from model.layers import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_head, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ffn, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention + Residual + Norm
        attn_out = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # FFN + Residual + Norm
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x

