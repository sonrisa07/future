import torch.nn as nn


class PreLayer(nn.Module):

    def __init__(self, d, hidden: list[int]):
        super(PreLayer, self).__init__()
        assert len(hidden) > 0
        self.net = nn.ModuleList()
        pre = d

        for x in hidden[:-1]:
            self.net.extend([
                nn.Linear(pre, x),
                nn.BatchNorm1d(x),
                nn.ReLU()
            ])
            pre = x

        self.net.append(nn.Linear(pre, hidden[-1]))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
