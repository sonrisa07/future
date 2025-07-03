import torch.nn as nn
from module.DCN import DCN
from module.DiffConv import DiffConv


class STBlock(nn.Module):

    def __init__(self, kernel, d, feature_dim, K):
        super(STBlock, self).__init__()
        self.tem_conv = DCN(feature_dim, kernel, d)
        self.spt_conv = DiffConv(feature_dim, K)
        self.ln = nn.LayerNorm(feature_dim)

    def forward(self, H, A):
        H_temp = self.tem_conv(H)
        H_spt = self.spt_conv(H_temp, A)
        H_out = self.ln(H_spt + H)
        return H_out
