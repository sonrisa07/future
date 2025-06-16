import torch.nn as nn

from module.DCN import DCN


class Mfstgcn(nn.Module):
    def __init__(self, feature_dim, tem_kernel, d):
        super(Mfstgcn, self).__init__()
        self.net = nn.ModuleList()
        depth = len(d)
        for i in range(depth):
            self.net.append(DCN(feature_dim, tem_kernel, d[i]))
            self.net.append()
