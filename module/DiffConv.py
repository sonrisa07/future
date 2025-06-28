import torch
import torch.nn as nn


class DiffConv(nn.Module):
    def __init__(self, dim, K):
        super(DiffConv, self).__init__()
        self.K = K

        self.weight_forward = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, dim)) for _ in range(K + 1)]
        )
        self.weight_backward = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, dim)) for _ in range(K + 1)]
        )
        self.bias = nn.Parameter(torch.zeros(dim))

        for w in self.weight_forward:
            nn.init.xavier_uniform_(w)
        for w in self.weight_backward:
            nn.init.xavier_uniform_(w)

    def forward(self, H, A):
        b, k, n, d = H.shape
        out_dim = self.weight_forward[0].shape[1]

        out = torch.zeros((b, k, n, out_dim), device=H.device)

        for t in range(k):
            print(A.shape)
            print(H.shape)
            print(t)
            A_t = A[t]
            H_t = H[:, t, :, :]

            tmp = 0
            for kk in range(self.K + 1):
                if kk == 0:
                    A_k = torch.eye(n, device=H.device)
                else:
                    A_k = torch.matrix_power(A_t, kk)
                A_k_T = A_k.transpose(0, 1)

                H_f = H_t @ self.weight_forward[kk]  # (b, n, out_dim)
                H_b = H_t @ self.weight_backward[kk]  # (b, n, out_dim)

                H_f = torch.einsum("ij,bjd->bid", A_k, H_f)
                H_b = torch.einsum("ij,bjd->bid", A_k_T, H_b)

                tmp += H_f + H_b

            out[:, t, :, :] = tmp

        if self.bias is not None:
            out += self.bias

        return out  # (b, k, n, out_dim)
