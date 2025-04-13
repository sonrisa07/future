import gc
from bisect import bisect_left, bisect_right

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rich.progress import track
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import haversine_distances
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric_temporal import STConv
from tqdm import tqdm

from model.QoSModel import QoSModel
from model.layers import FourierTemporalAttention, CoEncoderLayer, AutoEmbedding
from utils import convert_percentage_to_decimal, get_path

device0 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device1 = 'cuda:1' if torch.cuda.is_available() else 'cpu'
device1 = 'cpu'
lr = 1e-2
decay = 5e-4


def generate_graph(server_df, min_cluster_size, dis):
    Lon, Lat = server_df['lon'].values, server_df['lat'].values

    def haversine_meters(x, y=None):
        distance_rad = haversine_distances(x, y)
        return distance_rad * 6371000

    lat_rad = np.radians(Lat)
    lon_rad = np.radians(Lon)
    d_matrix = haversine_meters(np.c_[lat_rad, lon_rad])

    clustering = HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed').fit_predict(d_matrix)

    g_df = pd.DataFrame({
        'id': range(len(Lon)),
        'lon': Lon,
        'lat': Lat,
        'group': clustering
    })

    G = nx.Graph()

    for cluster_label in np.unique(clustering):
        if cluster_label == -1:
            continue

        cluster_points = g_df[g_df['group'] == cluster_label]

        core_points = cluster_points[cluster_points['id'].isin(np.where(clustering == cluster_label)[0])]

        for index, row in core_points.iterrows():
            G.add_node(row['id'], pos=(row['lon'], row['lat']))

        for core_index, core_point in core_points.iterrows():
            for _, other_point in core_points.iterrows():
                if core_point['id'] != other_point['id']:

                    core_lat_lon_rad = np.radians([core_point['lat'], core_point['lon']])
                    other_lat_lon_rad = np.radians([other_point['lat'], other_point['lon']])

                    distance = haversine_meters(np.array([core_lat_lon_rad, other_lat_lon_rad]))

                    if distance[0][1] <= dis:
                        G.add_edge(core_point['id'], other_point['id'], weight=distance)

    edges_x, edges_y = [], []
    for x in G.edges:
        edges_x.append([int(x[0])])
        edges_y.append([int(x[1])])

    return np.stack([np.hstack(edges_x), np.hstack(edges_y)])


class MyDataset(Dataset):

    def __init__(self,
                 user_df,
                 server_df,
                 load_df,
                 service_df,
                 inv_df,
                 k,
                 p):

        load_df['computing_load'] = load_df['computing_load'].apply(convert_percentage_to_decimal)
        load_df['storage_load'] = load_df['storage_load'].apply(convert_percentage_to_decimal)
        load_df['bandwidth_load'] = load_df['bandwidth_load'].apply(convert_percentage_to_decimal)

        user_df.sort_values(by=['timestamp', 'id'], inplace=True)
        load_df.sort_values(by=['timestamp', 'eid'], inplace=True)
        inv_df.sort_values(by=['timestamp', 'uid', 'eid', 'sid'], inplace=True)

        timestamps = sorted(user_df['timestamp'].unique())
        ids = sorted(user_df['id'].unique())

        timestamp_values = inv_df['timestamp'].values.astype(np.int32)
        indices = inv_df[['timestamp', 'uid', 'eid', 'sid']].values.astype(np.int32)
        rt_values = inv_df['rt'].values.astype(np.float32)

        inter_np = np.zeros((len(timestamps) - k - p + 1, k, len(server_df), len(ids)), dtype=np.int8)

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

        for b in range(len(timestamps) - k - p + 1):
            st = b + k
            ed = b + k + p - 1
            start_idx = bisect_left(timestamp_values, st)
            end_idx = bisect_right(timestamp_values, ed)
            block_indices = indices[start_idx:end_idx]
            block_values = rt_values[start_idx:end_idx]
            x = block_indices[:, 0] - st
            x = x.reshape(-1, 1)
            off_indices = np.concatenate((x, block_indices[:, 1:]), axis=-1)
            for i in range(len(off_indices)):
                inter_np[b, off_indices[i][0], off_indices[i][2], off_indices[i][1]] = 1
            block_sparse_qos = torch.sparse_coo_tensor(torch.tensor(off_indices.T),
                                                       torch.tensor(block_values),
                                                       size=(p, len(ids), len(server_df), len(service_df)))
            self.qos.append(block_sparse_qos)

        g = inv_df.groupby(['timestamp', 'uid'])['sid'].apply(list)

        svc = [[[0 for _ in range(len(service_df))] for _ in range(len(ids))] for _ in range(len(timestamps))]

        for (t, uid), slist in g.items():
            for sid in slist:
                svc[t][uid][sid] += 1

        svc_tensor = np.zeros((len(timestamps) - k - p + 1, k, len(ids), len(service_df)), dtype=np.float32)
        user_tensor = np.zeros((len(timestamps) - k - p + 1, k, len(ids), 4), dtype=np.float32)

        user_np = user_df[['lat', 'lon', 'speed', 'direction']].values.astype(np.float32)
        svc_np = np.array(svc, dtype=np.int32)

        for b in range(len(timestamps) - k - p + 1):
            svc_tensor[b] = svc_np[b:b + k]
            for t in range(k):
                user_tensor[b, t] = user_np[(b + t) * len(ids):(b + t + 1) * len(ids)]

        self.svc_tensor = torch.from_numpy(svc_tensor).permute(0, 2, 1, 3)
        self.user_tensor = torch.from_numpy(user_tensor).permute(0, 2, 1, 3)

        temp_df = pd.merge(inv_df, service_df, left_on='sid', right_on='sid')
        temp_df = pd.merge(load_df, temp_df, left_on=['timestamp', 'eid'], right_on=['timestamp', 'eid'], how='left')
        temp_df.fillna(0, inplace=True)
        temp_df = temp_df.groupby(['timestamp', 'eid'], as_index=False).agg({
            'computing': 'sum',
            'storage': 'sum',
            'bandwidth': 'sum'
        })

        temp_df.sort_values(['timestamp', 'eid'], inplace=True)

        load_np = load_df[['computing_load', 'storage_load', 'bandwidth_load']].values.astype(np.float32)
        srv_np = temp_df[['computing', 'storage', 'bandwidth']].values.astype(np.float32)

        srv_tensor = np.zeros((len(timestamps) - k - p + 1, k, len(server_df), 6), np.float32)

        for b in range(len(timestamps) - k - p + 1):
            for t in range(k):
                srv_tensor[b, t] = np.hstack((load_np[(b + t) * len(server_df):(b + t + 1) * len(server_df)],
                                              srv_np[(b + t) * len(server_df):(b + t + 1) * len(server_df)]))

        self.srv_tensor = torch.from_numpy(srv_tensor).permute(0, 2, 1, 3)

    def __getitem__(self, idx):
        return self.user_tensor[idx], self.svc_tensor[idx], self.srv_tensor[idx], self.inter_tensor[idx], self.qos[
            idx].to_dense()

    def __len__(self):
        return len(self.user_tensor)


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class NutNet(nn.Module):

    def __init__(self,
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

        self.usr_attr = usr_attr  # n * 1
        self.srv_attr = srv_attr  # m * 4
        self.svc_attr = svc_attr  # v * 4

        self.srv_attr[:, 1:] -= 1  # 1 ~ level -> 0 ~ level-1
        self.svc_attr[:, 1:] -= 1

        self.usr_emb = AutoEmbedding([usr_attr.shape[0]], emb_dim)
        self.srv_emb = AutoEmbedding([srv_attr.shape[0], level, level, level], emb_dim)
        self.svc_emb = AutoEmbedding([svc_attr.shape[0], level, level, level], emb_dim)

        self.user_proj = nn.Linear(5 * emb_dim + 4, d_model)
        self.srv_proj = nn.Linear(4 * emb_dim + 6, d_model)

        self.user_f_attn = FourierTemporalAttention(d_model)
        self.srv_f_attn = FourierTemporalAttention(d_model)

        self.encoder = CoEncoderLayer(d_model, head)

        self.tem_spa_net = STConv(srv_attr.shape[0], d_model, 2 * d_model, d_model, kernel_size, cheb_k)

    def forward(self, tra, u_inv, srv, mask, qos, edge_index):
        """
        :param tra: b * n * k * 4
        :param u_inv: b * n * k * v
        :param srv: b * m * k * 6
        :param mask: b * k * m * n
        :param qos: b * p * n * m * v (sparse tensor)
        :param edge_index: [2,
        """

        usr_emb = self.usr_emb(self.usr_attr)  # [n, emb_dim]
        srv_emb = self.srv_emb(self.srv_attr)  # [m, emb_dim * 4]
        svc_emb = self.svc_emb(self.svc_attr)  # [v, emb_dim * 4]

        b, n, m, v, k, p = tra.shape[0], usr_emb.shape[0], srv_emb.shape[0], svc_emb.shape[0], tra.shape[-2], qos.shape[
            1]

        usr_mat = (usr_emb.unsqueeze(0).unsqueeze(2)
                   .expand(b, n, k, usr_emb.shape[-1]))  # [b, n, k, emb_dim]
        u_inv = u_inv @ svc_emb  # [b, n, k, emb_dim * 4]
        usr_mat = torch.concat((usr_mat, tra, u_inv), dim=-1)  # [b, n, k, emb_dim * 5 + 4]

        srv_mat = (srv_emb.unsqueeze(0).unsqueeze(2)
                   .expand(b, m, k, srv_emb.shape[-1]))  # [b, m, k, emb_dim * 4]
        srv_mat = torch.concat((srv_mat, srv), dim=-1)  # [b, m, k, emb * 4 + 6]

        usr_mat = self.user_proj(usr_mat)  # [b, n, k, d_model]
        srv_mat = self.srv_proj(srv_mat)  # [b, m, k, d_model]

        usr_mat = self.user_f_attn(usr_mat, usr_mat, usr_mat)  # [b, n, k, d_model]
        srv_mat = self.srv_f_attn(srv_mat, srv_mat, srv_mat)  # [b, m, k, d_model]

        usr_mat = usr_mat.transpose(-2, -3)  # [b, k, n, d_model]
        srv_mat = srv_mat.transpose(-2, -3)  # [b, k, m, d_model]

        tem_srv = self.encoder(srv_mat, usr_mat, usr_mat, mask)  # [b, k, m, d_model]

        tem_srv = self.tem_spa_net(tem_srv, edge_index).squeeze(1)  # [b, m, d_model]

        usr_mat = usr_mat[:, -1, ...]  # [b, n, d_model]

        svc_emb = svc_emb.unsqueeze(0).expand(b, -1, -1)  # [b, v, emb_dim * 4]

        return usr_mat, tem_srv, svc_emb

    def to(self, device):
        super(NutNet, self).to(device0)
        self.usr_attr = self.usr_attr.to(device0)
        self.srv_attr = self.srv_attr.to(device0)
        self.svc_attr = self.svc_attr.to(device0)
        return self


class DownTask(nn.Module):

    def __init__(self, emb_dim, d_model, p):
        super(DownTask, self).__init__()
        self.usr_srv = nn.ModuleList([
            LinearBlock(d_model * 2, d_model),
            LinearBlock(d_model, 32),
            LinearBlock(32, 16),
            LinearBlock(16, 8),
        ])
        self.srv_svc = nn.ModuleList([
            LinearBlock(d_model + emb_dim * 4, 32),
            LinearBlock(32, 16),
            LinearBlock(16, 8),
        ])
        self.qos_net = nn.ModuleList([
            LinearBlock(16, 4),
            LinearBlock(4, p)
        ])

    def forward(self, usr_mat, tem_srv, svc_emb):
        n, m, v = usr_mat.shape[1], tem_srv.shape[1], svc_emb.shape[1]
        usr_mat = usr_mat.unsqueeze(2).expand(-1, -1, m, -1)  # [b, n, m, d_model]
        tem_srv1 = tem_srv.unsqueeze(1).expand(-1, n, -1, -1)  # [b, n, m, d_model]
        x1 = torch.concat((usr_mat, tem_srv1), dim=-1)
        for net in self.usr_srv:
            x1 = net(x1)
        tem_srv2 = tem_srv.unsqueeze(2).expand(-1, -1, v, -1)  # [b, m, v, d_model]
        svc_emb = svc_emb.unsqueeze(1).expand(-1, m, -1, -1)  # [b, m, v, emb_dim * 4]
        x2 = torch.concat((tem_srv2, svc_emb), dim=-1)
        for net in self.srv_svc:
            x2 = net(x2)
        x1 = x1.to(device1)
        x2 = x2.to(device1)
        x1 = x1.unsqueeze(3).expand(-1, -1, -1, v, -1)
        x2 = x2.unsqueeze(1).expand(-1, n, -1, -1, -1)
        x = torch.concat((x1, x2), dim=-1)
        for net in self.qos_net:
            x = net(x)
        return x

    def to(self, device0, device1):
        self.usr_srv = self.usr_srv.to(device0)
        self.srv_svc = self.srv_svc.to(device0)
        self.qos_net = self.qos_net.to(device1)
        return self


class Nut:
    def __init__(self,
                 user_df,
                 server_df,
                 load_df,
                 service_df,
                 inv_df,
                 k):
        super(Nut, self).__init__()
        self.dataset = MyDataset(user_df, server_df, load_df, service_df, inv_df, 5, 1)

        usr_attr = torch.arange(user_df['id'].nunique()).view(-1, 1)
        svc_attr = torch.from_numpy(service_df[['sid', 'computing', 'storage', 'bandwidth']].values.astype(np.int32))
        srv_attr = torch.from_numpy(server_df[['eid', 'computing', 'storage', 'bandwidth']].values.astype(np.int32))

        self.net = [
            NutNet(user_df['id'].nunique(), usr_attr, srv_attr, svc_attr,
                   k, 1, 16, 4, 256, 3, 3, 5),
            DownTask(8, 64, 1)
        ]
        self.edge_index = torch.LongTensor(generate_graph(server_df, 3, 600)).to(device0)

    def get_dataloaders(self, scope, split):
        data_size = len(self.dataset)
        train_valid_size = int(scope * data_size)
        test_size = data_size - train_valid_size

        train_size = int(split * train_valid_size)
        valid_size = train_valid_size - train_size

        train_dataset, valid_dataset, test_dataset = random_split(self.dataset, [train_size, valid_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=4, num_workers=4,
                                  shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, num_workers=4,
                                 shuffle=False)

        return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    user_df = pd.read_csv(get_path('users.csv'))
    load_df = pd.read_csv(get_path('loads.csv'))
    server_df = pd.read_csv(get_path('servers.csv'))
    service_df = pd.read_csv(get_path('services.csv'))
    inv_df = pd.read_csv(get_path('invocation.csv'))

    dataset = MyDataset(user_df, server_df, load_df, service_df, inv_df, 5, 1)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset, 0.2, 0.8)

    usr_attr = torch.arange(user_df['id'].nunique()).view(-1, 1)
    svc_attr = torch.from_numpy(service_df[['sid', 'computing', 'storage', 'bandwidth']].values.astype(np.int32))
    srv_attr = torch.from_numpy(server_df[['eid', 'computing', 'storage', 'bandwidth']].values.astype(np.int32))

    edge_index = torch.LongTensor(generate_graph(server_df, 3, 600)).to(device0)

    model = NutNet(user_df['id'].nunique(), usr_attr, srv_attr, svc_attr,
                   5, 1, 8, 4, 64, 3, 3, 5)
    down_model = DownTask(8, 64, 1)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=decay,
    )
    model = model.to(device0)
    down_model = down_model.to(device0, device1)
    print(len(train_loader))
    for tra, u_inv, srv, inter, qos in track(train_loader):
        tra, u_inv, srv, inter = tra.to(device0), u_inv.to(device0), srv.to(device0), inter.to(device0)
        usr_mat, tem_srv, svc_emb = model(tra, u_inv, srv, inter, qos, edge_index)
        qos = qos.to(device1)
        qos = qos.permute(0, 2, 3, 4, 1)
        preds = down_model(usr_mat, tem_srv, svc_emb)
        qos = qos.contiguous().view(-1, 1)
        preds = preds.contiguous().view(-1, 1)
        mask = qos != 0
        loss = criterion(preds[mask], qos[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.detach().item())
