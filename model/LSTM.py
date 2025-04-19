from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader

from model.QoSModel import QoSModel
from utils import StandardScaler
from utils import convert_percentage_to_decimal
import torch.nn as nn
import torch.nn.functional as F


class MyDataset(Dataset):

    def __init__(self,
                 user_df,
                 server_df,
                 load_df,
                 service_df,
                 inv_df,
                 k):
        load_df['computing_load'] = load_df['computing_load'].apply(convert_percentage_to_decimal)
        load_df['storage_load'] = load_df['storage_load'].apply(convert_percentage_to_decimal)
        load_df['bandwidth_load'] = load_df['bandwidth_load'].apply(convert_percentage_to_decimal)

        user_df.sort_values(by=['uid', 'timestamp'], inplace=True)

        grouped = user_df.groupby('uid')

        tra_window = dict()

        for uid, g in grouped:
            data = g[['lat', 'lon', 'speed', 'direction']].values
            for i in range(len(data) - k):
                tra_window.setdefault(uid, []).append(data[i: i + k])

        grouped = load_df.groupby('eid')

        load_window = dict()

        for eid, g in grouped:
            data = g[['computing_load', 'storage_load', 'bandwidth_load']].values
            for i in range(len(data) - k):
                load_window.setdefault(eid, []).append(data[i: i + k])

        temp = []
        for i in range(len(server_df)):
            for j in range(0, user_df['timestamp'].max() + 1):
                temp.append([i, j])

        t_df = pd.DataFrame(temp, columns=['eid', 'timestamp'])

        t_df = pd.merge(t_df, inv_df, on=['eid', 'timestamp'], how='left')

        t_df = t_df.groupby(['eid', 'timestamp'], as_index=False).agg({
            'sid': lambda x: x.dropna().astype(int).tolist()
        }).reset_index(drop=True)

        grouped = t_df.groupby('eid')

        svc_window = dict()

        for eid, g in grouped:
            data = g['sid'].tolist()
            lst = []
            for i in range(len(data)):
                count = Counter(data[i])
                result = [0.] * len(service_df)
                for key, value in count.items():
                    result[key] = value
                lst.append(result)
            data = np.array(lst)
            for i in range(len(data) - k):
                svc_window.setdefault(eid, []).append(data[i: i + k])

        inv_df = inv_df[inv_df['timestamp'] >= k]
        uids = inv_df['uid'].values
        eids = inv_df['eid'].values
        sids = inv_df['sid'].values
        ts = inv_df['timestamp'].values
        qos = inv_df[['rt']].values

        max_time = np.max(ts) - k

        self.k = k
        self.tra = []
        self.info = []
        self.load = []
        self.svc = []
        self.q = []

        for uid in range(user_df['uid'].nunique()):
            self.tra.append(tra_window[uid])

        for eid in range(server_df['eid'].nunique()):
            self.load.append(load_window[eid])

        for sid in range(service_df['sid'].nunique()):
            self.svc.append(svc_window[sid])

        for i in range(len(inv_df)):
            uid = uids[i]
            eid = eids[i]
            sid = sids[i]
            t = ts[i]
            self.info.append([uid, eid, sid, t])
            self.q.append(qos[i])

        self.tra = np.array(self.tra, dtype=np.float32)
        self.info = np.array(self.info, dtype=np.int32)
        self.load = np.array(self.load, dtype=np.float32)
        self.svc = np.array(self.svc, dtype=np.float32)
        self.q = np.array(self.q, dtype=np.float32)

        self.scaler = StandardScaler(mean=self.q[:, 0].mean(), std=self.q[:, 0].std())

        self.tra = torch.from_numpy(self.tra)
        self.info = torch.from_numpy(self.info)
        self.q = torch.from_numpy(self.q)
        self.load = torch.from_numpy(self.load)
        self.svc = torch.from_numpy(self.svc)

    def __getitem__(self, idx):
        uid, eid, sid, t = self.info[idx]
        t_idx = t - self.k
        return self.tra[uid][t_idx], self.info[idx], self.load[eid][t_idx], self.svc[sid][t_idx], self.q[idx]

    def __len__(self):
        return self.info.shape[0]


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return self.relu(x)


class NutNet(nn.Module):

    def __init__(self, load_level, k, srv_res, svc_res, tra_dim=8, emb_dim=8, emb_load_dim=16):
        super(NutNet, self).__init__()
        self.k = k
        self.tra_net = nn.LSTM(4, tra_dim)
        self.e_emb = nn.Embedding(load_level * 3 + 1, emb_dim)
        self.s_emb = nn.Embedding(load_level * 3 + 1, emb_dim)
        self.e_load = nn.Sequential(
            nn.Linear(3 + 3 * emb_dim, emb_load_dim),
            nn.Linear(emb_load_dim, emb_load_dim // 2)
        )
        self.load_net = nn.LSTM(emb_load_dim // 2 + 3 * emb_dim, 56)
        self.qos_net = nn.ModuleList([
            LinearBlock(88, 64),
            LinearBlock(64, 32),
            LinearBlock(32, 16),
            LinearBlock(16, 1),
        ])

        self.srv_res = torch.tensor(srv_res)
        self.svc_res = torch.tensor(svc_res)

    def forward(self, tra, info, load, svc, edge_index):
        """
        :param tra: (b, k, 4)
        :param info: (b, 3)
        :param load: (b, k, 3)
        :param svc: (b, k, N_svc)
        :param edge_index: []
        :return: (2,)
        """

        srv_res_emb = self.e_emb(self.srv_res)  # N_srv * 3 * emb_dim
        svc_res_emb = self.s_emb(self.svc_res)  # N_svc * 3 * emb_dim
        srv_res_emb = srv_res_emb.contiguous().view(-1, 3 * self.e_emb.embedding_dim)  # N_srv * (3*emb_dim)
        svc_res_emb = svc_res_emb.contiguous().view(-1, 3 * self.e_emb.embedding_dim)  # N_svc * (3*emb_dim)

        # *********** 轨迹预测 ***********
        tra, _ = self.tra_net(tra)  # b * k * 8

        tra = tra[:, -1, :]  # b * 8
        # *******************************

        # ****** 服务器资源embedding ******
        e_rate = srv_res_emb[info[:, 1]]  # b * (3*emb_dim)
        # *******************************

        # **** 服务器资源 + 服务器负载 ****
        e_load = self.e_load(torch.concat((load, e_rate.unsqueeze(1).expand(-1, self.k, -1)),
                                          -1))  # b * k * (3 + 3 * emb_dim) -> b * k * emb_load_dim // 2

        e_load = F.relu(e_load)
        # *******************************

        # **********服务供给总和**********
        svc_tot = svc @ svc_res_emb  # b * k * (3*emb_dim)
        # *******************************

        # *******服务需求embedding********
        s_rate = svc_res_emb[info[:, 2]]  # b * (3*emb_dim)
        # *******************************

        # ********服务器时空预测**********
        e_load = torch.concat((e_load, svc_tot), -1)  # b * k * (emb_load_dim // 2 + 3 * emb_dim)

        e_load, _ = self.load_net(e_load)  # b * k * 56

        e_load = e_load[:, -1, :]  # b * 56
        # *******************************

        # *************QoS预测************
        x = torch.concat((tra, e_load, s_rate), -1)
        for net in self.qos_net:
            x = net(x)
        # *******************************

        return x

    def to(self, device):
        super(NutNet, self).to(device)
        self.srv_res = self.srv_res.to(device)
        self.svc_res = self.svc_res.to(device)
        return self


class LSTM(QoSModel):
    def __init__(self,
                 user_df,
                 server_df,
                 load_df,
                 service_df,
                 inv_df,
                 k):
        super().__init__(k)
        self.dataset = MyDataset(user_df, server_df, load_df, service_df, inv_df, k)
        self._scaler = self.dataset.scaler
        srv_res = np.stack(
            (server_df['computing'].values, server_df['storage'].values + 5, server_df['bandwidth'].values + 10),
            axis=-1)
        svc_res = np.stack(
            (service_df['computing'].values, service_df['storage'].values + 5, service_df['bandwidth'].values + 10),
            axis=-1)
        self._net = NutNet(server_df['computing'].nunique(), k, srv_res, svc_res)
        self._edge_index = np.array([])

    def get_dataloaders(self, scope, split):
        data_size = len(self.dataset)
        train_valid_size = int(scope * data_size)
        test_size = data_size - train_valid_size

        train_size = int(split * train_valid_size)
        valid_size = train_valid_size - train_size

        train_dataset, valid_dataset, test_dataset = random_split(self.dataset, [train_size, valid_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=2048, num_workers=4,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=2048, num_workers=4,
                                  shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=2048, num_workers=4,
                                 shuffle=False)

        return train_loader, valid_loader, test_loader

    @property
    def edge_index(self):
        return self._edge_index

    @property
    def net(self):
        return self._net

    @property
    def scaler(self):
        return self._scaler
