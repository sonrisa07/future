from collections import Counter
import random
from rich.progress import track

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, random_split, DataLoader, Subset
from torch_geometric_temporal import STConv

from model.layers import AutoEmbedding
from module import DCN.DCN
from utils import StandardScaler, mercator, meters_to_mercator_unit, sort_dataset
from utils import convert_percentage_to_decimal, get_path
import torch.nn as nn
import torch.nn.functional as F


class MyDataset(IterableDataset):
    def __init__(
            self,
            user_df: pd.DataFrame,
            server_df: pd.DataFrame,
            load_df: pd.DataFrame,
            service_df: pd.DataFrame,
            inv_df: pd.DataFrame,
            k: int,
            shuffle
    ):
        self.shuffle = shuffle
        inv_df.sort_values(by=["timestamp", "eid"], inplace=True)
        inv_d = {}
        inv_np = inv_df[["timestamp", "uid", "eid", "sid", "rt"]].to_numpy()
        for i in range(len(inv_np)):
            t = inv_np[i][0].item()
            if t not in inv_d:
                inv_d[t] = []
            inv_d[t].append(tuple(inv_np[i][1:]))

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
        srv_np = server_df[
            ["eid", "lat", "lon", "radius"]
        ].to_numpy()
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

        load_df['computing_load'] = load_df['computing_load'].apply(convert_percentage_to_decimal)
        load_df['storage_load'] = load_df['storage_load'].apply(convert_percentage_to_decimal)
        load_df['bandwidth_load'] = load_df['bandwidth_load'].apply(convert_percentage_to_decimal)
        load_np = load_df[["eid", "timestamp", "computing_load", "storage_load", "bandwidth_load"]].to_numpy()
        load_d = {}
        for i in range(len(load_np)):
            eid, t = load_np[i][0], load_np[i][1]
            load_d[(t, eid)] = load_np[i][2:]

        t_list = []
        t_l_list = []
        t_len = int(inv_df["timestamp"].max()) + 1
        for t in range(t_len):
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
        print(eh_srv_np.shape)

        user_np = user_df[
            ["timestamp", "uid", "lat", "lon", "speed", "direction"]
        ].to_numpy()
        user_t_d = {}
        for i in range(len(user_np)):
            t, uid = user_np[i][0].item(), user_np[i][1].item()
            user_t_d[(t, uid)] = user_np[i][1:]

        service_df.sort_values("sid", inplace=True)
        svc_np = service_df[["sid", "computing", "storage", "bandwidth"]].to_numpy()
        self.svc_tensor = torch.from_numpy(svc_np)

        self.srv_tensor = torch.from_numpy(server_df[["eid", "computing", "storage", "bandwidth"]].to_numpy())

        self.end_d_user = {}
        self.end_d_info = {}
        self.end_d_rt = {}
        self.edge_tensor = []
        self.load_tensor = []
        for st in range(t_len - k):
            en = st + k
            self.edge_tensor.append(eh_srv_np[st : st + k])
            self.load_tensor.append(load_np[st : st + k])
            temp_u = []
            temp_i = []
            temp_r = []
            for uid, eid, sid, rt in inv_d[en]:
                temp = []
                for i in range(st, en):
                    temp.append(user_t_d[(i, uid)])
                temp_u.append(np.stack(temp, axis=0))
                temp_i.append(np.array([eid, sid]))
                temp_r.append(rt)
            self.end_d_user[st] = torch.from_numpy(np.stack(temp_u, axis=0))
            self.end_d_info[st] = torch.from_numpy(np.stack(temp_i, axis=0))
            self.end_d_rt[st] = torch.from_numpy(np.stack(temp_r, axis=0))

        self.edge_tensor = np.stack(self.edge_tensor, axis=0)
        self.load_tensor = np.stack(self.load_tensor, axis=0)

        print(self.edge_tensor.shape)
        print(self.load_tensor.shape)
        print(len(self.end_d_user))
        print(len(self.end_d_info))
        print(len(self.end_d_rt))

    def __iter__(self):
        idxes = list(range(len(self.edge_tensor)))
        if self.shuffle:
            random.shuffle(idxes)
        for t in idxes:
            yield (
                self.edge_tensor[t],
                self.load_tensor[t],
                self.end_d_user[t],
                self.end_d_info[t],
                self.end_d_rt[t],
            )

    def get_ext_data(self):
        return self.srv_tensor, self.svc_tensor


class DynamicNet(nn.Module):
    def __init__(self, n_user, n_server, n_service, srv_level, svc_level, k, emb_dim, edge_d, tra_hidden,):
        super(DynamicNet, self).__init__()
        self.user_emb = AutoEmbedding([n_user], emb_dim)
        self.srv_emb = AutoEmbedding([n_server, srv_level, srv_level, srv_level], emb_dim)
        self.svc_emb = AutoEmbedding([n_service, svc_level, svc_level, svc_level], emb_dim)
        self.lstm = nn.LSTM(4, tra_hidden)
        
        self.E1 = nn.Parameter(torch.empty(edge_d, edge_d, edge_d))
        self.E2 = nn.Parameter(torch.empty(k, edge_d))
        self.proj_edge = nn.Linear(11, edge_d)

        nn.init.xavier_uniform_(self.E1)
        nn.init.normal_(self.E2, mean=0, std=0.01)

        self.tem_gcn = DCN()

        

    def forward(self, srv_attr, svc_attr, edge, load, tra, info):
        """
        :param srv_attr: N_e * 4
        :param svc_attr: N_s * 4
        :param edge: k * N_e * 11 [lat, lon, radius,...(8)]
        :param load: k * N_e * 3
        :param tra: b * k * 4 [lat, lon, speed, direction]
        :param info: b * k * 2 [eid, sid]
        """

        srv_emb = self.srv_emb(srv_attr) # [N_e, emb_dim * 4]
        svc_emb = self.svc_emb(svc_attr) # [n_s, emb_dim * 4]

        tra = self.lstm(tra)[:, -1, :] # [b, tra_hidden]
        edge = self.proj_edge(edge) # [k, n_e, edge_d]
        
        A_tmp = torch.einsum('ouv,to,tiu,tjv->tij', self.E1, self.E2, edge, edge)
        A_pos = F.relu(A_tmp)
        A = torch.softmax(A_pos, dim=-1) # [k, N_e, N_e]

        load = load.unsqueeze(0);
        load = 



class Dynamic:
    def __init__(self, user_df, server_df, load_df, service_df, inv_df, k):
        self.dataset = MyDataset(
            user_df,
            server_df,
            load_df,
            service_df,
            inv_df,
            k,
            True
        )
        self.srv, self.svc = self.dataset.get_ext_data()

    def train_one_epoch(self):
        pass


if __name__ == "__main__":
    Dynamic(
        pd.read_csv(get_path("user.csv")),
        pd.read_csv(get_path("server.csv")),
        pd.read_csv(get_path("load.csv")),
        pd.read_csv(get_path("service.csv")),
        pd.read_csv(get_path("enhance_invocation.csv")),
        5,
    )
