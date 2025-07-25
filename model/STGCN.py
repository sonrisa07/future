from collections import Counter

import networkx as nx
from model.layers import AutoEmbedding
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader, Subset
from torch_geometric_temporal import STConv

from model.STModel import STModel
from module.PreLayer import PreLayer
from utils import StandardScaler, mercator, meters_to_mercator_unit, sort_dataset
from utils import convert_percentage_to_decimal, get_path
import torch.nn as nn
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, user_df, server_df, load_df, service_df, inv_df, k):
        inv_df.sort_values(by=["timestamp", "uid", "eid", "sid"], inplace=True)

        load_df["computing_load"] = load_df["computing_load"].apply(
            convert_percentage_to_decimal
        )
        load_df["storage_load"] = load_df["storage_load"].apply(
            convert_percentage_to_decimal
        )
        load_df["bandwidth_load"] = load_df["bandwidth_load"].apply(
            convert_percentage_to_decimal
        )

        user_df.sort_values(by=["uid", "timestamp"], inplace=True)

        grouped = user_df.groupby("uid")

        tra_window = dict()

        for uid, g in grouped:
            data = g[["lat", "lon", "speed", "direction"]].values
            for i in range(len(data) - k):
                tra_window.setdefault(uid, {})[i + k] = data[i : i + k]

        load_df.sort_values(by="timestamp", inplace=True)
        grouped = load_df.groupby("timestamp")
        sorted_grouped = grouped.apply(lambda x: x.sort_values(by="eid"))

        load = []
        for st in range(0, len(grouped) - k):
            window = sorted_grouped[sorted_grouped["timestamp"].between(st, st + k - 1)]
            window = np.stack(
                [
                    window[window["timestamp"] == t][
                        ["computing_load", "storage_load", "bandwidth_load"]
                    ].to_numpy()
                    for t in range(st, st + k)
                ]
            )
            load.append(window)

        temp = []
        for i in range(len(server_df)):
            for j in range(0, user_df["timestamp"].max() + 1):
                temp.append([i, j])

        t_df = pd.DataFrame(temp, columns=["eid", "timestamp"])

        t_df = pd.merge(t_df, inv_df, on=["eid", "timestamp"], how="left")

        t_df = (
            t_df.groupby(["timestamp", "eid"], as_index=False)
            .agg({"sid": lambda x: x.dropna().astype(int).tolist()})
            .reset_index(drop=True)
        )

        t_df.sort_values(by="timestamp", inplace=True)
        grouped = t_df.groupby("timestamp")

        temp = []
        for t, g in grouped:
            g.sort_values(by="eid", inplace=True)
            data = g["sid"].tolist()
            lst = []
            for i in range(len(data)):
                count = Counter(data[i])
                result = [0.0] * len(service_df)
                for key, value in count.items():
                    result[key] = value
                lst.append(result)
            temp.append(np.stack(lst, axis=0))
        temp = np.stack(temp, axis=0)

        svc = []
        for st in range(0, len(grouped) - k):
            svc.append(temp[st : st + k, ...])

        self.tra = []
        self.info = []
        self.q = []
        self.load = load
        self.svc = svc

        inv_df.drop_duplicates(subset=["timestamp", "uid", "eid", "sid"], inplace=True)
        inv_df = inv_df[inv_df["timestamp"] >= k]
        uids = inv_df["uid"].values
        eids = inv_df["eid"].values
        sids = inv_df["sid"].values
        ts = inv_df["timestamp"].values
        qos = inv_df[["rt"]].values
        for i in range(len(inv_df)):
            uid = uids[i]
            eid = eids[i]
            sid = sids[i]
            t = ts[i]
            self.tra.append(tra_window[uid][t])
            self.info.append([uid, eid, sid, t])
            self.q.append(qos[i])

        self.tra = np.array(self.tra, dtype=np.float32)
        self.info = np.array(self.info, dtype=np.int32)
        self.q = np.array(self.q, dtype=np.float32)
        self.load = np.array(self.load, dtype=np.float32)
        self.svc = np.array(self.svc, dtype=np.float32)

        self.scaler = StandardScaler(mean=self.q[:, 0].mean(), std=self.q[:, 0].std())

        self.tra = torch.from_numpy(self.tra)
        self.info = torch.from_numpy(self.info)
        self.q = torch.from_numpy(self.q)
        self.load = torch.from_numpy(self.load)
        self.svc = torch.from_numpy(self.svc)

    def __getitem__(self, idx):
        return self.tra[idx], self.info[idx], self.q[idx]

    def __len__(self):
        return self.info.shape[0]


class NutNet(nn.Module):
    def __init__(self, k, srv_res, svc_res, tra_dim=16, emb_dim=8, emb_load_dim=16):
        super(NutNet, self).__init__()
        self.k = k
        self.tra_net = nn.LSTM(5 + emb_dim, tra_dim)
        self.u_emb = AutoEmbedding([1000], emb_dim)
        self.e_emb = AutoEmbedding([len(srv_res), 7, 7, 7], emb_dim)
        self.s_emb = AutoEmbedding([len(svc_res), 5, 5, 5], emb_dim)
        self.e_load = nn.Sequential(
            nn.Linear(3 + 3 * emb_dim, emb_load_dim),
            nn.Linear(emb_load_dim, emb_load_dim // 2),
        )
        self.load_net = nn.ModuleList(
            [
                STConv(srv_res.shape[0], emb_load_dim // 2 + 3 * emb_dim, 64, 32, 3, 3),
                STConv(srv_res.shape[0], 32, 64, 56, 3, 3),
            ]
        )
        self.qos_net = PreLayer(64 + 3 * emb_dim, [32, 16, 1])

        self.srv_res = srv_res
        self.svc_res = svc_res

    def forward(self, tra, info, load, svc, edge_index):
        """
        :param tra: (batch, k, 5)
        :param info: (batch, 4)
        :param load: (T, k, N_srv, 3)
        :param svc: (T, k, N_srv, N_svc)
        :param edge_index: (2, edge_number)
        :return: (2,)
        """

        usr_emb = self.u_emb(info[:, 0])  # [b, emb_dim]

        # *********** 轨迹预测 ***********
        tra, _ = self.tra_net(torch.concat((tra, usr_emb), dim=-1))  # b * k * tra_dim

        tra = tra[:, -1, :]  # b * tra_dim
        # *******************************

        unique_values, inverse_indices = torch.unique(info[:, 3], return_inverse=True)
        pos = (unique_values - load.shape[1]).to("cpu")
        load, svc = load[pos].to(tra.device), svc[pos].to(tra.device)

        # ****** 服务器资源embedding ******
        srv_emb = self.e_emb(self.srv_res[:, :4])  # [m, emb_dim * 4]

        # *******************************

        # **** 服务器资源 + 服务器负载 ****
        # *******************************

        # *******服务需求embedding********
        # *******************************

        # ********服务器时空预测**********
        # *******************************

        # *************QoS预测************
        # *******************************

        return x

    def to(self, device):
        super(NutNet, self).to(device)
        self.srv_res = self.srv_res.to(device)
        self.svc_res = self.svc_res.to(device)
        return self


class STGCN(STModel):
    def __init__(self, user_df, server_df, load_df, service_df, inv_df, k):
        super().__init__(k)
        self.dataset = MyDataset(user_df, server_df, load_df, service_df, inv_df, k)
        self._scaler = self.dataset.scaler
        self.srv_res = np.stack(
            (
                server_df["eid"].values,
                server_df["lat"].values,
                server_df["lon"].values,
                server_df["radius"].values,
                server_df["computing"].values,
                server_df["storage"].values,
                server_df["bandwidth"].values,
            ),
            axis=-1,
        )
        self.svc_res = np.stack(
            (
                service_df["sid"].values,
                service_df["computing"].values,
                service_df["storage"].values,
                service_df["bandwidth"].values,
            ),
            axis=-1,
        )
        self.srv_res = torch.from_numpy(self.srv_res)
        self.svc_res = torch.from_numpy(self.svc_res)
        self.feature_enhance()
        self._edge_index = torch.LongTensor(pd.read_csv(get_path("edges.csv")).values.T)
        self._net = NutNet(k, self.srv_res, self.svc_res)

    def get_dataloaders(self, scope, split):
        data_size = len(self.dataset)
        train_valid_size = int(scope * data_size)
        test_size = data_size - train_valid_size

        train_size = int(split * train_valid_size)
        valid_size = train_valid_size - train_size

        train_dataset = Subset(self.dataset, list(range(0, train_size)))
        valid_dataset = Subset(
            self.dataset, list(range(train_size, train_size + valid_size))
        )
        test_dataset = Subset(
            self.dataset, list(range(train_size + valid_size, data_size))
        )

        train_dataset = sort_dataset(train_dataset, self.dataset, 1, 3)
        valid_dataset = sort_dataset(valid_dataset, self.dataset, 1, 3)
        test_dataset = sort_dataset(test_dataset, self.dataset, 1, 3)

        train_loader = DataLoader(
            train_dataset, batch_size=128, num_workers=4, shuffle=False
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=128, num_workers=4, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=128, num_workers=4, shuffle=False
        )

        return train_loader, valid_loader, test_loader

    def feature_enhance(self):
        n = self.dataset.tra.shape[0]
        k = self.dataset.tra.shape[1]
        self.dataset.tra = self.dataset.tra.reshape(n * k, 4)
        self.dataset.tra[:, 0:2] = mercator(
            self.dataset.tra[:, 0], self.dataset.tra[:, 1]
        )
        self.srv_res[:, 1:3] = mercator(self.srv_res[:, 1], self.srv_res[:, 2])
        all_geo = torch.concat((self.dataset.tra[:, 0:2], self.srv_res[:, 1:3]), dim=0)
        all_min, _ = torch.min(all_geo, dim=0)
        all_max, _ = torch.max(all_geo, dim=0)
        eps = 1e-8
        (self.dataset.tra[:, 0:2] - all_min) / (all_max - all_min + eps)
        (self.srv_res[:, 1:3] - all_min) / (all_max - all_min + eps)
        col = self.srv_res[:, 3]
        norm_col = (col - col.min()) / (col.max() - col.min())
        self.srv_res[:, 3] = norm_col
        self.dataset.tra = self.dataset.tra.reshape(n, k, 4)
        vx = self.dataset.tra[..., 2] * torch.cos(self.dataset.tra[..., 3])
        vy = self.dataset.tra[..., 2] * torch.sin(self.dataset.tra[..., 3])
        self.dataset.tra = torch.concat(
            (
                self.dataset.tra[..., 0:2],
                vx.unsqueeze(-1),
                vy.unsqueeze(-1),
                self.dataset.tra[..., -1].unsqueeze(-1),
            ),
            dim=-1,
        )

    def get_tsp_data(self):
        return self.dataset.load, self.dataset.svc

    @property
    def net(self):
        return self._net

    @property
    def edge_index(self):
        return self._edge_index

    @property
    def scaler(self):
        return self._scaler
