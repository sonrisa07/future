import gc
import random
from bisect import bisect_left, bisect_right

import torch.nn.functional as F
import networkx as nx
import numpy as np
import pandas as pd
import torch
from rich.progress import track
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import haversine_distances
from torch import nn
from torch.nn import LSTM
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric_temporal import STConv
from tqdm import tqdm

from model.QoSModel import QoSModel
from model.get import getSvc
from model.layers import FourierTemporalAttention, AutoEmbedding
from module.CoDecoder import CoDecoder
from module.Encoder import Encoder
from module.Fnet import FNet
from module.PreLayer import PreLayer
from utils import sort_dataset, positional_encoding, mercator, meters_to_mercator_unit
from utils import convert_percentage_to_decimal, get_path


def generate_graph(server_df, min_cluster_size, dis):
    Lon, Lat = server_df["lon"].values, server_df["lat"].values

    def haversine_meters(x, y=None):
        distance_rad = haversine_distances(x, y)
        return distance_rad * 6371000

    lat_rad = np.radians(Lat)
    lon_rad = np.radians(Lon)
    d_matrix = haversine_meters(np.c_[lat_rad, lon_rad])

    clustering = HDBSCAN(
        min_cluster_size=min_cluster_size, metric="precomputed"
    ).fit_predict(d_matrix)

    g_df = pd.DataFrame(
        {"id": range(len(Lon)), "lon": Lon, "lat": Lat, "group": clustering}
    )

    G = nx.Graph()

    for cluster_label in np.unique(clustering):
        if cluster_label == -1:
            continue

        cluster_points = g_df[g_df["group"] == cluster_label]

        core_points = cluster_points[
            cluster_points["id"].isin(np.where(clustering == cluster_label)[0])
        ]

        for index, row in core_points.iterrows():
            G.add_node(row["id"], pos=(row["lon"], row["lat"]))

        for core_index, core_point in core_points.iterrows():
            for _, other_point in core_points.iterrows():
                if core_point["id"] != other_point["id"]:
                    core_lat_lon_rad = np.radians(
                        [core_point["lat"], core_point["lon"]]
                    )
                    other_lat_lon_rad = np.radians(
                        [other_point["lat"], other_point["lon"]]
                    )

                    distance = haversine_meters(
                        np.array([core_lat_lon_rad, other_lat_lon_rad])
                    )

                    if distance[0][1] <= dis:
                        G.add_edge(core_point["id"], other_point["id"], weight=distance)

    edges_x, edges_y = [], []
    for x in G.edges:
        edges_x.append([int(x[0])])
        edges_y.append([int(x[1])])

    return np.stack([np.hstack(edges_x), np.hstack(edges_y)])


class MyDataset(Dataset):
    def __init__(self, user_df, server_df, load_df, service_df, inv_df, k, p):
        load_df["computing_load"] = load_df["computing_load"].apply(
            convert_percentage_to_decimal
        )
        load_df["storage_load"] = load_df["storage_load"].apply(
            convert_percentage_to_decimal
        )
        load_df["bandwidth_load"] = load_df["bandwidth_load"].apply(
            convert_percentage_to_decimal
        )

        inv_df.drop_duplicates(subset=["timestamp", "uid", "eid", "sid"], inplace=True)

        user_df.sort_values(by=["timestamp", "uid"], inplace=True)
        load_df.sort_values(by=["timestamp", "eid"], inplace=True)
        inv_df.sort_values(by=["timestamp", "uid", "eid", "sid"], inplace=True)

        timestamps = sorted(user_df["timestamp"].unique())
        ids = sorted(user_df["uid"].unique())

        timestamp_values = inv_df["timestamp"].values.astype(np.int32)
        indices = inv_df[["timestamp", "uid", "eid", "sid"]].values.astype(np.int32)
        rt_values = inv_df["rt"].values.astype(np.float32)

        inter_np = np.zeros(
            (len(timestamps) - k - p + 1, k, len(server_df), len(ids)), dtype=np.int8
        )

        for b in range(len(timestamps) - k - p + 1):
            st = b
            ed = b + k - 1
            start_idx = bisect_left(timestamp_values, st)
            end_idx = bisect_right(timestamp_values, ed)
            block_indices = indices[start_idx:end_idx]
            x = block_indices[:, 0] - st
            x = x.reshape(-1, 1)
            off_indices = np.concatenate((x, block_indices[:, 1:]), axis=-1)
            for i in range(len(off_indices)):
                inter_np[b, off_indices[i][0], off_indices[i][2], off_indices[i][1]] = 1

        self.inter_tensor = torch.from_numpy(inter_np)

        self.qos = []

        g = inv_df.groupby(["timestamp", "uid"])["sid"].apply(list)

        svc = [
            [[0 for _ in range(len(service_df))] for _ in range(len(ids))]
            for _ in range(len(timestamps))
        ]

        for (t, uid), slist in g.items():
            for sid in slist:
                svc[t][uid][sid] += 1

        svc_tensor = np.zeros(
            (len(timestamps) - k - p + 1, k, len(ids), len(service_df)),
            dtype=np.float32,
        )
        user_tensor = np.zeros(
            (len(timestamps) - k - p + 1, k, len(ids), 4), dtype=np.float32
        )

        user_np = user_df[["lat", "lon", "speed", "direction"]].values.astype(
            np.float32
        )
        svc_np = np.array(svc, dtype=np.int32)

        for b in range(len(timestamps) - k - p + 1):
            svc_tensor[b] = svc_np[b : b + k]
            for t in range(k):
                user_tensor[b, t] = user_np[(b + t) * len(ids) : (b + t + 1) * len(ids)]

        self.svc_tensor = torch.from_numpy(svc_tensor).permute(0, 2, 1, 3)
        self.user_tensor = torch.from_numpy(user_tensor).permute(0, 2, 1, 3)

        e_svc = [
            [[0 for _ in range(len(service_df))] for _ in range(len(server_df))]
            for _ in range(len(timestamps))
        ]

        g = inv_df.groupby(["timestamp", "eid"])["sid"].apply(list)
        for (t, eid), slist in g.items():
            for sid in slist:
                e_svc[t][eid][sid] += 1

        e_svc_np = np.array(e_svc, dtype=np.int32)

        load_np = load_df[
            ["computing_load", "storage_load", "bandwidth_load"]
        ].values.astype(np.float32)

        srv_tensor = np.zeros(
            (len(timestamps) - k - p + 1, k, len(server_df), 3), np.float32
        )
        e_svc_tensor = np.zeros(
            (len(timestamps) - k - p + 1, k, len(server_df), len(service_df)),
            dtype=np.float32,
        )

        for b in range(len(timestamps) - k - p + 1):
            e_svc_tensor[b] = e_svc_np[b : b + k]
            for t in range(k):
                srv_tensor[b, t] = load_np[
                    (b + t) * len(server_df) : (b + t + 1) * len(server_df)
                ]

        self.e_svc_tensor = torch.from_numpy(e_svc_tensor).permute(0, 2, 1, 3)
        self.srv_tensor = torch.from_numpy(srv_tensor).permute(0, 2, 1, 3)

        self.chunking = []

        inv_df = inv_df[inv_df["timestamp"] >= k]
        uids = inv_df["uid"].values
        eids = inv_df["eid"].values
        sids = inv_df["sid"].values
        ts = inv_df["timestamp"].values
        rts = inv_df[["rt"]].values
        for i in range(len(inv_df)):
            self.chunking.append(
                torch.tensor((uids[i], eids[i], sids[i], ts[i]), dtype=torch.int32)
            )
            self.qos.append(rts[i])

        self.qos = torch.from_numpy(np.array(self.qos, dtype=np.float32))

    def __getitem__(self, idx):
        return self.chunking[idx], self.qos[idx]

    def __len__(self):
        return len(self.chunking)


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_features)
        # self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return self.relu(x)


class NutNet(nn.Module):
    def __init__(
        self,
        n,
        usr_attr,
        srv_attr,
        svc_attr,
        k,
        p,
        emb_dim,
        head,
        d_model,
        kernel_size,
        cheb_k,
        level,
    ):
        super(NutNet, self).__init__()

        self.usr_attr = usr_attr  # n * 1  uid
        self.srv_attr = (
            srv_attr  # m * 7  eid, computing, storage, bandwidth, lat, lon, radius
        )
        self.svc_attr = svc_attr  # v * 4  sid, computing, storage, bandwidth

        self.srv_attr[:, 1:4] -= 1  # 1 ~ level -> 0 ~ level-1
        self.svc_attr[:, 1:] -= 1

        self.usr_emb = AutoEmbedding([usr_attr.shape[0]], emb_dim)
        self.svc_emb = AutoEmbedding([svc_attr.shape[0], level, level, level], emb_dim)
        self.srv_emb = AutoEmbedding([srv_attr.shape[0], level, level, level], emb_dim)

        self.load_proj = nn.Linear(3 + 4 * emb_dim, d_model)
        self.fuse_proj = nn.Linear(d_model * 3, d_model)

        self.usr_proj = nn.Linear(emb_dim * 4 + d_model // 2, d_model)
        self.srv_proj = nn.Linear(3, d_model)

        self.usr_norm = nn.BatchNorm2d(k)
        self.srv_norm = nn.BatchNorm2d(k)

        self.tra_proj = nn.Linear(5 + emb_dim, d_model // 2)
        # self.tra_gru = nn.GRU(d_model // 2, d_model // 2, batch_first=True)
        self.tra_lstm = nn.LSTM(d_model // 2, d_model)
        self.tra_attn = Encoder(d_model // 2, d_model, 4, 1, 0.2)
        # self.usr_f_attn = Encoder(emb_dim * 2, emb_dim * 4, head, 2, 0.0)
        # self.srv_f_attn = FNet(d_model, 2, 0.2)
        # self.usr_f_attn = LSTM(4, emb_dim * 2)
        # self.srv_f_attn = LSTM(3, emb_dim * 2)

        self.decoder = CoDecoder(d_model, 2 * d_model, head, 2, 0.2)

        self.load_net = nn.Linear(emb_dim * 4 + 3, d_model)

        self.tem_spa_net = nn.ModuleList(
            [
                STConv(
                    srv_attr.shape[0],
                    d_model,
                    2 * d_model,
                    d_model,
                    kernel_size,
                    cheb_k,
                ),
                STConv(
                    srv_attr.shape[0],
                    d_model,
                    2 * d_model,
                    d_model,
                    kernel_size,
                    cheb_k,
                ),
            ]
        )

        self.qos_net = PreLayer(d_model // 2 + d_model + emb_dim * 4, [32, 16, 8, p])

    def forward(self, tra, u_inv, srv, e_inv, mask, info, qos, edge_index):
        """
        :param tra: T * n * k * 5
        :param u_inv: T * n * k * v
        :param srv: T * m * k * 3
        :param e_inv: T * m * k * v
        :param mask: T * k * m * n
        :param info: b * 4
        :param qos: b * p
        :param edge_index: [2,
        """

        n, m, k, p = tra.shape[1], srv.shape[1], tra.shape[2], qos.shape[1]

        usr_emb = self.usr_emb(self.usr_attr)  # [n, emb_dim]
        svc_emb = self.svc_emb(self.svc_attr)  # [v, emb_dim * 4]
        srv_emb = self.srv_emb(self.srv_attr[:, :4].int())  # [m, emb_dim * 4]

        unique_values, inverse_indices = torch.unique(info[:, 3], return_inverse=True)
        t = len(unique_values)

        tra = tra[unique_values - k]
        srv = srv[unique_values - k]
        mask = mask[unique_values - k]

        unique_values = unique_values.to("cpu")

        u_inv = u_inv[unique_values - k]
        e_inv = e_inv[unique_values - k]

        u_inv = u_inv.to(tra.device)
        e_inv = e_inv.to(tra.device)

        usr_mat = (
            usr_emb.unsqueeze(0).unsqueeze(2).expand(t, -1, k, -1)
        )  # [t, n, k, emb_dim]
        tra = torch.concat((tra, usr_mat), dim=-1)  # [t, n, k, emb_dim + 5]
        tra = self.tra_proj(tra)  # [t, n, k, d_model // 2]
        tra, _ = self.tra_lstm(tra)
        tra = tra[:, :, -1, :]  # [t, n, d_model // 2]

        srv_mat = torch.concat(
            (srv_emb.unsqueeze(0).unsqueeze(2).expand(t, -1, k, -1), srv), dim=-1
        )  # [t, m, k, emb_dim * 4 + 3]

        tem_srv = self.load_net(srv_mat)  # [t, m, k, d_model]

        for net in self.tem_spa_net:
            tem_srv = net(tem_srv, edge_index)

        tem_srv = tem_srv.squeeze(1)  # [t, m, d_model]
        tra = tra[inverse_indices, info[:, 0]]  # [b, d_model // 2]

        tem_srv = tem_srv[inverse_indices, info[:, 1]]  # [b, d_model]

        svc_emb = svc_emb[info[:, 2]]  # [b, emb_dim * 4]

        x = torch.concat(
            (tra, tem_srv, svc_emb), dim=-1
        )  # [b, d_model // 2 + d_model + emb_dim * 4]

        x = self.qos_net(x)

        return x

        # tra = tra.contiguous().view(t * n, k, -1)
        # tra, _ = self.tra_gru(tra)  # [t * n, k, d_model]
        #
        # tra = tra.view(t, n, k, -1)  # [t, n, k, d_model]
        #
        # srv_mat = (
        #     self.srv_attr[:, -3:].unsqueeze(0).unsqueeze(1).expand(t, k, -1, -1)
        # )  # [t, k, m, 3]
        #
        # u_inv = (u_inv @ svc_emb).transpose(-2, -3)  # [t, k, n, emb_dim * 4]
        # usr_mat = torch.concat(
        #     (tra.transpose(-2, -3), u_inv), dim=-1
        # )  # [t, k, n, emb_dim * 4 + d_model // 2]
        #
        # usr_mat = self.usr_proj(usr_mat)  # [t, k, n, d_model]
        # srv_mat = self.srv_proj(srv_mat)  # [t, k, m, d_model]
        #
        # tem_srv = self.decoder(usr_mat, srv_mat, mask)  # [t, k, m, d_model]
        #
        # srv = self.load_proj(
        #     torch.concat(
        #         (srv, srv_emb.unsqueeze(0).unsqueeze(2).expand(t, -1, k, -1)), dim=-1
        #     )
        # )  # [t, m, k, d_model]
        #
        # tem_srv = torch.concat(
        #     (tem_srv, srv.transpose(-2, -3), srv_mat), dim=-1
        # )  # [t, k, m, d_model * 3]
        #
        # tem_srv = self.fuse_proj(tem_srv)  # [t, k, m, d_model]
        #
        # tem_srv = self.srv_norm(tem_srv)  # [t, k, m, d_model]
        #
        # for net in self.tem_spa_net:
        #     tem_srv = net(tem_srv, edge_index)
        #
        # tem_srv = tem_srv.squeeze(1)  # [t, m, d_model]
        #
        # tra = tra[inverse_indices, info[:, 0], -1]  # [b, d_model]
        # tem_srv = tem_srv[inverse_indices, info[:, 1]]  # [b, d_model]
        # svc_emb = svc_emb[info[:, 2]]  # [b, emb_dim * 4]
        #
        # x = torch.concat(
        #     (tra, tem_srv, svc_emb), dim=-1
        # )  # [b, d_model + d_model + emb_dim * 4]
        # x = self.qos_net(x)

        return x

    def to(self, device):
        super(NutNet, self).to(device)
        self.usr_attr = self.usr_attr.to(device)
        self.srv_attr = self.srv_attr.to(device)
        self.svc_attr = self.svc_attr.to(device)
        return self


class Real:
    def __init__(self, user_df, server_df, load_df, service_df, inv_df, k):
        super(Real, self).__init__()
        self.dataset = MyDataset(user_df, server_df, load_df, service_df, inv_df, k, 1)

        usr_attr = torch.arange(user_df["uid"].nunique()).view(-1, 1)
        srv_attr = torch.from_numpy(
            server_df[
                ["eid", "computing", "storage", "bandwidth", "lat", "lon", "radius"]
            ].values.astype(np.float32)
        )
        svc_attr = torch.from_numpy(
            service_df[["sid", "computing", "storage", "bandwidth"]].values.astype(
                np.int32
            )
        )

        self.feature_enhance(usr_attr, srv_attr, svc_attr)

        self.net = NutNet(
            user_df["uid"].nunique(),
            usr_attr,
            srv_attr,
            svc_attr,
            k,
            1,
            8,
            4,
            32,
            3,
            3,
            6,
        )
        self.edge_index = torch.LongTensor(pd.read_csv(get_path("edges.csv")).values.T)

    def get_dataloaders(self, scope, split):
        data_size = len(self.dataset)
        train_valid_size = int(scope * data_size)
        test_size = data_size - train_valid_size

        train_size = int(split * train_valid_size)
        valid_size = train_valid_size - train_size

        train_dataset, valid_dataset, test_dataset = random_split(
            self.dataset, [train_size, valid_size, test_size]
        )

        train_dataset = sort_dataset(train_dataset, self.dataset, 0, 3)
        valid_dataset = sort_dataset(valid_dataset, self.dataset, 0, 3)
        test_dataset = sort_dataset(test_dataset, self.dataset, 0, 3)

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

    def get_tsp_data(self):
        return (
            self.dataset.user_tensor,
            self.dataset.svc_tensor,
            self.dataset.srv_tensor,
            self.dataset.e_svc_tensor,
            self.dataset.inter_tensor,
        )

    def feature_enhance(self, usr_attr, srv_attr, svc_attr):
        self.dataset.user_tensor[..., 0:2] = mercator(
            self.dataset.user_tensor[..., 0:2]
        )
        srv_attr[:, 4:-1] = mercator(srv_attr[:, 4:-1])

        all_geo = torch.concat(
            (
                self.dataset.user_tensor[..., 0:2].reshape(-1, 2),
                srv_attr[..., 4:-1].reshape(-1, 2),
            ),
            dim=0,
        )
        min_all, _ = torch.min(all_geo, dim=0)
        max_all, _ = torch.max(all_geo, dim=0)

        mercator_per_meter = meters_to_mercator_unit(1.0, 50)
        mercator_range = (max_all - min_all).max()

        self.dataset.user_tensor[..., 0:2] = (
            self.dataset.user_tensor[..., 0:2] - min_all
        ) / (max_all - min_all + 1e-8)
        srv_attr[:, 4:-1] = (srv_attr[:, 4:-1] - min_all) / (max_all - min_all + 1e-8)
        srv_attr[:, -1] = srv_attr[:, -1] * mercator_per_meter / (mercator_range + 1e-8)

        vx = self.dataset.user_tensor[..., -2] * torch.cos(
            self.dataset.user_tensor[..., -1]
        )
        vy = self.dataset.user_tensor[..., -2] * torch.sin(
            self.dataset.user_tensor[..., -1]
        )

        self.dataset.user_tensor = torch.concat(
            (
                self.dataset.user_tensor[..., 0:2],
                vy.unsqueeze(-1),
                vx.unsqueeze(-1),
                self.dataset.user_tensor[..., -1].unsqueeze(-1),
            ),
            dim=-1,
        )
