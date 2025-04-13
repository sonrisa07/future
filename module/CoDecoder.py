import torch.nn as nn

from module.DecoderLayer import DecoderLayer


class CoDecoder(nn.Module):

    def __init__(self, d_model, d_ffn, n_head, depth, dropout, k=None):
        super(CoDecoder, self).__init__()
        self.net = nn.ModuleList([
            DecoderLayer(d_model, d_ffn, n_head, dropout, k) for _ in range(depth)
        ])

    def forward(self, usr_mat, srv_mat, mask):
        x = srv_mat
        for net in self.net:
            x = net(usr_mat, x, mask)
        return x
