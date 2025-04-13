from torch import nn

from module.FNetLayer import FNetLayer


class FNet(nn.Module):

    def __init__(self, d_model, depth, dropout):
        super(FNet, self).__init__()
        self.net = nn.ModuleList([
            FNetLayer(d_model, dropout) for _ in range(depth)
        ])

    def forward(self, x):
        for net in self.net:
            x = net(x)
        return x
