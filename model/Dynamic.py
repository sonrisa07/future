from collections import Counter
import random

from torch.nn.utils.rnn import pad_sequence

from module.Mfstgcn import Mfstgcn
from module.PreLayer import PreLayer
from rich.progress import track

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader, Subset

from model.layers import AutoEmbedding
from utils import convert_percentage_to_decimal, get_path, sort_dataset
import torch.nn as nn
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(
            self,
            user_df: pd.DataFrame,
            server_df: pd.DataFrame,
            load_df: pd.DataFrame,
            service_df: pd.DataFrame,
            inv_df: pd.DataFrame,
            k: int,
    ):
        inv_df.sort_values(by=["timestamp", "eid"], inplace=True)
        inv_d = {}
        inv_np = inv_df[["uid", "eid", "sid", "rt", "timestamp"]].to_numpy()
        for i in range(len(inv_np)):
            t = inv_np[i][-1].item()
            if t not in inv_d:
                inv_d[t] = []
            inv_d[t].append(tuple(inv_np[i]))

        dynamic_features = (
            inv_df.drop_duplicates(["timestamp", "eid", "uid"], inplace=False)
            .groupby(by=["timestamp", "eid"], as_index=False)
            .agg(
                {
                    "link_distance": "mean",
                    "radial_speed": "mean",
                    "tangential_speed": "mean",
                    "relative_bearing": "mean",
                    "normalized_range": "mean",
                    "heading_offset": "mean",
                    "edge_margin": "mean",
                    "uid": "count",
                }
            )
        ).rename(columns={"uid": "cnt"})
        server_df.sort_values("eid", inplace=True)
        srv_np = server_df[["eid", "lat", "lon", "radius"]].to_numpy()
        srv_d = {}
        for i in range(len(server_df)):
            srv_d[srv_np[i][0].item()] = srv_np[i][1:]
        srv_t_d = {}
        for row in dynamic_features.itertuples(index=False):
            eid = row.eid
            t = row.timestamp
            srv_t_d[(t, eid)] = np.array(
                [
                    row.link_distance,
                    row.radial_speed,
                    row.tangential_speed,
                    row.relative_bearing,
                    row.normalized_range,
                    row.heading_offset,
                    row.edge_margin,
                    row.cnt,
                ]
            )

        load_df["computing_load"] = load_df["computing_load"].apply(
            convert_percentage_to_decimal
        )
        load_df["storage_load"] = load_df["storage_load"].apply(
            convert_percentage_to_decimal
        )
        load_df["bandwidth_load"] = load_df["bandwidth_load"].apply(
            convert_percentage_to_decimal
        )
        load_np = load_df[
            ["eid", "timestamp", "computing_load", "storage_load", "bandwidth_load"]
        ].to_numpy()
        load_d = {}
        for i in range(len(load_np)):
            eid, t = load_np[i][0], load_np[i][1]
            load_d[(t, eid)] = load_np[i][2:]

        t_list = []
        t_l_list = []
        t_len = int(inv_df["timestamp"].max()) + 1
        for t in track(range(t_len)):
            temp = []
            temp_l = []
            for eid in range(len(srv_np)):
                dynamic = srv_t_d.get((t, eid), np.array([0.0 for _ in range(8)]))
                temp.append(np.hstack((srv_d[eid], dynamic)))
                temp_l.append(load_d[(t, eid)])
            t_list.append(np.stack(temp, axis=0))
            t_l_list.append(np.stack(temp_l, axis=0))
        eh_srv_np = np.stack(t_list, axis=0)
        load_np = np.stack(t_l_list, axis=0)

        n_u = user_df["uid"].nunique()
        user_np = user_df[
            ["timestamp", "uid", "lat", "lon", "speed", "direction"]
        ].to_numpy()
        tra_d = {}
        for i in range(len(user_np)):
            uid, t = user_np[i][1], user_np[i][0]
            tra_d[(t, uid)] = user_np[i, 2:]

        t_list = []
        for t in track(range(t_len)):
            temp = []
            for uid in range(n_u):
                temp.append(tra_d[(t, uid)])
            t_list.append(np.stack(temp, axis=0))
        tra_np = np.stack(t_list, axis=0)

        svc_map = service_df.set_index("sid")[["computing", "storage", "bandwidth"]]
        merged = inv_df[["timestamp", "eid", "sid"]].copy()
        merged = merged.merge(svc_map, left_on="sid", right_index=True, how="left")
        agg = merged.groupby(["eid", "timestamp"])[
            ["computing", "storage", "bandwidth"]
        ].sum()

        svc_tot_d = {}
        for (eid, timestamp), row in agg.iterrows():
            svc_tot_d[(int(eid), int(timestamp))] = row.values

        n_e = server_df["eid"].nunique()
        t_list = []
        for t in track(range(t_len)):
            temp = []
            for eid in range(n_e):
                temp.append(svc_tot_d.get((eid, t), np.zeros(3, np.int32)))
            t_list.append(np.stack(temp, axis=0))
        svc_tot_np = np.stack(t_list, axis=0)

        self.end_d_info = {}
        self.end_d_rt = {}
        self.edge_tensor = []
        self.load_tensor = []
        self.tra_tensor = []
        self.svc_tot_tensor = []
        for st in track(range(t_len - k)):
            en = st + k
            self.edge_tensor.append(eh_srv_np[st: st + k])
            self.load_tensor.append(load_np[st: st + k])
            self.tra_tensor.append(tra_np[st: st + k])
            self.svc_tot_tensor.append(svc_tot_np[st: st + k])
            temp_i = []
            temp_r = []
            for uid, eid, sid, rt, tim in inv_d[en]:
                temp_i.append(np.array([uid, eid, sid, tim], dtype=np.int32))
                temp_r.append(rt)
            self.end_d_info[st] = torch.from_numpy(
                np.stack(temp_i, axis=0, dtype=np.int32)
            )
            self.end_d_rt[st] = torch.from_numpy(
                np.stack(temp_r, axis=0, dtype=np.float32)
            )

        self.edge_tensor = torch.from_numpy(
            np.stack(self.edge_tensor, axis=0, dtype=np.float32)
        )
        self.load_tensor = torch.from_numpy(
            np.stack(self.load_tensor, axis=0, dtype=np.float32)
        )
        self.tra_tensor = torch.from_numpy(
            np.stack(self.tra_tensor, axis=0, dtype=np.float32)
        )
        self.svc_tot_tensor = torch.from_numpy(
            np.stack(self.svc_tot_tensor, axis=0, dtype=np.float32)
        )

        print(self.tra_tensor.shape)  # [frame, k, N_u, 4]
        print(self.edge_tensor.shape)  # [frame, k, N_e, 11]
        print(self.load_tensor.shape)  # [frame, k, N_e, 3]
        print(self.svc_tot_tensor.shape)  # [frame, k, n_e, 3]
        print(len(self.end_d_info))  # [frame, n, 4]
        print(len(self.end_d_rt))  # [frame, n, 1]

        self.tra_tensor_chunk = []
        self.edge_tensor_chunk = []
        self.load_tensor_chunk = []
        self.svc_tot_tensor_chunk = []
        self.end_info_chunk = []
        self.end_d_rt_chunk = []
        batch_size = 256
        for t in range(self.tra_tensor.shape[0]):
            seq_k = self.end_d_info[t].shape[0] // batch_size
            tim = self.end_d_info[t][0][-1]
            for i in range(seq_k):
                self.tra_tensor_chunk.append(self.tra_tensor[t])
                self.edge_tensor_chunk.append(self.edge_tensor[t])
                self.load_tensor_chunk.append(self.load_tensor[t])
                self.svc_tot_tensor_chunk.append(self.svc_tot_tensor[t])
                self.end_info_chunk.append(self.end_d_info[t][i * batch_size: (i + 1) * batch_size])
                self.end_d_rt_chunk.append(self.end_d_rt[t][i * batch_size: (i + 1) * batch_size])
            rem = self.end_d_info[t].shape[0] - seq_k * batch_size
            if rem > 0:
                self.tra_tensor_chunk.append(self.tra_tensor[t])
                self.edge_tensor_chunk.append(self.edge_tensor[t])
                self.load_tensor_chunk.append(self.load_tensor[t])
                self.svc_tot_tensor_chunk.append(self.svc_tot_tensor[t])
                self.end_info_chunk.append(self.end_d_info[t][seq_k * batch_size: self.end_d_info[t].shape[0]])
                self.end_d_rt_chunk.append(self.end_d_rt[t][seq_k * batch_size: self.end_d_info[t].shape[0]])

        self.tra_tensor = torch.stack(self.tra_tensor_chunk, dim=0)
        self.edge_tensor = torch.stack(self.edge_tensor_chunk, dim=0)
        self.load_tensor = torch.stack(self.load_tensor_chunk, dim=0)
        self.svc_tot_tensor = torch.stack(self.svc_tot_tensor_chunk, dim=0)
        self.end_info_tensor = pad_sequence(
            self.end_info_chunk,
            batch_first=True,
            padding_value=0,
        )
        self.end_d_rt_tensor = pad_sequence(
            self.end_d_rt_chunk,
            batch_first=True,
            padding_value=0,
        )

        print(self.tra_tensor.shape)  # [N, k, N_u, 4]
        print(self.edge_tensor.shape)  # [N, k, N_e, 11]
        print(self.load_tensor.shape)  # [N, k, N_e, 3]
        print(self.svc_tot_tensor.shape)  # [N, k, n_e, 3]
        print(self.end_info_tensor.shape)  # [N, b, 4]
        print(self.end_d_rt_tensor.shape)  # [N, b, 1]

    def __getitem__(self, idx):
        return (
            self.tra_tensor[idx],  # [k, N_u, 4]
            self.edge_tensor[idx],  # [k, N_e, 11]
            self.load_tensor[idx],  # [k, N_e, 3]
            self.svc_tot_tensor[idx],  # [k, N_e, 3]
            self.end_info_tensor[idx],  # [b, 4]
            self.end_d_rt_tensor[idx],  # [b, 1]
        )

    def __len__(self):
        return len(self.edge_tensor)


class DynamicNet(nn.Module):
    def __init__(
            self,
            n_user,
            n_server,
            n_service,
            srv_attr,
            svc_attr,
            srv_level,
            svc_level,
            k,
            emb_dim=8,
            edge_d=8,
            tra_hidden=16,
            feature_dim=32,
            tem_kernel=3,
            d=None,
    ):
        super(DynamicNet, self).__init__()

        if d is None:
            d = [1, 2, 1, 2]
        self.srv_attr = srv_attr
        self.svc_attr = svc_attr

        self.usr_emb = AutoEmbedding([n_user], emb_dim)
        self.srv_emb = AutoEmbedding(
            [n_server, srv_level, srv_level, srv_level], emb_dim
        )
        self.svc_emb = AutoEmbedding(
            [n_service, svc_level, svc_level, svc_level], emb_dim
        )
        self.lstm = nn.LSTM(emb_dim + 4, tra_hidden, batch_first=True)

        self.E1_p = nn.Parameter(nn.init.xavier_uniform_(torch.empty(edge_d, edge_d, edge_d)))
        self.E2_p = nn.Parameter(nn.init.normal_(torch.empty(k, edge_d), 0., 0.01))
        self.proj_edge_p = nn.Linear(11, edge_d)

        self.E1_a = nn.Parameter(nn.init.xavier_uniform_(torch.empty(edge_d, edge_d, edge_d)))
        self.E2_a = nn.Parameter(nn.init.normal_(torch.empty(k, edge_d), 0., 0.01))
        self.proj_edge_a = nn.Linear(11, edge_d)

        self.proj_p = nn.Linear(3, feature_dim)
        self.proj_a = nn.Linear(3, feature_dim)

        self.mfstgcn = Mfstgcn(feature_dim, tem_kernel, d, 2)

        self.qos_net = PreLayer(
            tra_hidden + emb_dim * 4 * 2 + feature_dim, [64, 32, 16, 8, 4, 1]
        )

        self.clip = 5.0

    def forward(self, edge, load, svc_tot, tra, info):
        """
        :param edge: B * k * N_e * 11 [lat, lon, radius,...(8)]
        :param load: B * k * N_e * 3
        :param svc_tot: B * k * N_e * 3
        :param tra: B * k * N_u * 4 [uid, lat, lon, speed, direction]
        :param info: B * b * 2 [uid, eid, sid, tim]
        """

        edge = edge.squeeze(0)
        load = load.squeeze(0)
        svc_tot = svc_tot.squeeze(0)
        tra = tra.squeeze(0)
        info = info.squeeze(0)

        k, n_u = tra.shape[0], tra.shape[1]
        usr_emb = self.usr_emb(
            torch.arange(n_u, dtype=torch.int, device=tra.device).reshape(-1, 1)
        )  # [N_u, emb_dim]
        srv_emb = self.srv_emb(self.srv_attr)  # [N_e, emb_dim * 4]
        svc_emb = self.svc_emb(self.svc_attr)  # [n_s, emb_dim * 4]

        tra = torch.concat(
            (usr_emb.unsqueeze(0).expand(k, -1, -1), tra), dim=-1
        )  # [k, N_u, emb_dim + 4]
        tra = tra.transpose(0, 1)  # [N_u, k, emb_dim + 4]
        tra, _ = self.lstm(tra)
        tra = tra[:, -1, :]  # [N_u, tra_hidden]
        edge_p = self.proj_edge_p(edge)  # [k, n_e, edge_d]
        edge_a = self.proj_edge_a(edge)  # [k, n_e, edge_d]

        A_p_tmp = torch.einsum(
            "ouv,to,tiu,tjv->tij", self.E1_p, self.E2_p, edge_p, edge_p
        )
        A_p_pos = F.relu(A_p_tmp)
        A_p = torch.softmax(A_p_pos, dim=-1)  # [k, N_e, N_e]

        A_a_tmp = torch.einsum(
            "ouv,to,tiu,tjv->tij", self.E1_a, self.E2_a, edge_a, edge_a
        )
        A_a_pos = F.relu(A_a_tmp)
        A_a = torch.softmax(A_a_pos, dim=-1)  # [k, N_e, N_e]

        load = load.unsqueeze(0)  # [1, k, N_e, 3]
        svc_tot = svc_tot.unsqueeze(0)  # [1, k, N_e, 3]

        load = self.proj_p(load)
        svc_tot = self.proj_a(svc_tot)

        srv_fea = self.mfstgcn(load, svc_tot, A_p, A_a)  # depth * [1, N_e, feature_dim]

        srv_fea = torch.stack(srv_fea, dim=1)  # [1, depth, N_e, feature_dim]

        srv_fea = torch.mean(srv_fea, dim=1).squeeze(0)  # [N_e, feature_dim]

        srv_fea = torch.concat(
            (srv_fea[info[:, 1]], srv_emb[info[:, 1]]), dim=-1
        )  # [b, feature_dim + emb_dim * 4]

        svc_fea = svc_emb[info[:, 2]]  # [b, emb_dim * 4]

        tra_fea = tra[info[:, 0]]  # [b, tra_hidden]

        qos = self.qos_net(torch.concat((tra_fea, srv_fea, svc_fea), dim=-1))

        return qos

    def to(self, device):
        super(DynamicNet, self).to(device)
        self.srv_attr = self.srv_attr.to(device)
        self.svc_attr = self.svc_attr.to(device)
        return self


class Dynamic:
    def __init__(self, user_df, server_df, load_df, service_df, inv_df, k):
        self.dataset = MyDataset(user_df, server_df, load_df, service_df, inv_df, k)
        srv_attr = torch.from_numpy(
            server_df[["eid", "computing", "storage", "bandwidth"]].values.astype(
                np.int32
            )
        )
        svc_attr = torch.from_numpy(
            service_df[["sid", "computing", "storage", "bandwidth"]].values.astype(
                np.int32
            )
        )
        self.net = DynamicNet(
            user_df["uid"].nunique(),
            server_df["eid"].nunique(),
            service_df["sid"].nunique(),
            srv_attr,
            svc_attr,
            7,
            5,
            k,
        )

        self.t_boundary = inv_df["timestamp"].max() + 1

    def get_dataloaders(self, scope, split):
        train_valid_tim = int(scope * self.t_boundary)
        train_tim = int(split * train_valid_tim)

        train_idx = []
        valid_idx = []
        test_idx = []

        for idx in track(range(len(self.dataset))):
            t = int(self.dataset[idx][-2][0][-1])
            if t < train_tim:
                train_idx.append(idx)
            elif train_tim <= t < train_valid_tim:
                valid_idx.append(idx)
            else:
                test_idx.append(idx)

        print(train_tim, train_valid_tim, self.t_boundary)
        train_dataset = Subset(self.dataset, train_idx)
        valid_dataset = Subset(self.dataset, valid_idx)
        test_dataset = Subset(self.dataset, test_idx)
        print("train_dataset size: {}".format(len(train_dataset)))
        print("valid_dataset size: {}".format(len(valid_dataset)))
        print("test_dataset size: {}".format(len(test_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        return train_loader, valid_loader, test_loader
