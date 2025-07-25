import torch.nn as nn

from module.STBlock import STBlock
from module.DCN import DCN


class Mfstgcn(nn.Module):
    def __init__(self, feature_dim, tem_kernel, d, K):
        super(Mfstgcn, self).__init__()
        self.net_p = nn.ModuleList()
        self.net_a = nn.ModuleList()
        self.prop = nn.Linear(feature_dim, feature_dim, bias=False)
        nn.init.xavier_uniform_(self.prop.weight)
        self.depth = len(d)
        for i in range(self.depth):
            self.net_p.append(STBlock(tem_kernel, d[i], feature_dim, K))
            self.net_a.append(STBlock(tem_kernel, d[i], feature_dim, K))

    def forward(self, H_p, H_a, A_p, A_a):
        out = []
        for i in range(self.depth):
            H_p = self.net_p[i](H_p, A_p)
            H_a = self.net_a[i](H_a, A_a)
            out.append(H_p[:, -1, :, :])
            H_p = H_p + self.prop(H_a)
        return out
