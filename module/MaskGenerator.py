import torch
import torch.nn as nn


class MaskGenerator(nn.Module):

    def __init__(self, d_model, n_u):
        super(MaskGenerator, self).__init__()
        self.gru_u = nn.GRU(d_model, n_u, batch_first=True)

    def forward(self, user):
        t, n, k, d = user.shape
        uf = user.view(t * n, k, d)
        hu, _ = self.gru_u(uf)
