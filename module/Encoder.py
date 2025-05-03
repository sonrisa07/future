import torch.nn as nn

from module.EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, d_model, d_ffn, n_head, depth, dropout):
        super(Encoder, self).__init__()
        self.net = nn.ModuleList([
            EncoderLayer(d_model, d_ffn, n_head, dropout) for _ in range(depth)
        ])

    def forward(self, x, mask):
        for net in self.net:
            x = net(x, mask)
        return x

