import torch.nn as nn
from module.DCN import DCN
from module.DiffConv import DiffConv

class STBlock(nn.Module):

    def __init__(self, kernel, d, feature_dim, K):
        super(STBlock, self).__init__()
        self.tem_conv = DCN(feature_dim, kernel, d)
        self.spt_conv = DiffConv(feature_dim, K)

    def forward(self, H, A):
        H = self.tem_conv(H)
        H = self.spt_conv(H, A)
        return H