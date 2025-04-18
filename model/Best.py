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
from module.Fnet import FNet
from module.PreLayer import PreLayer
from utils import sort_dataset, positional_encoding
from utils import convert_percentage_to_decimal, get_path, find_servers, haversine


# device0 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device1 = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# device1 = 'cpu'
# lr = 1e-2
# decay = 5e-4

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

        # inv_df.drop_duplicates(subset=['timestamp', 'uid', 'eid', 'sid'], inplace=True)

        df = pd.merge(left=inv_df, right=user_df, left_on=['uid', 'timestamp'], right_on=['uid', 'timestamp'])
        df = pd.merge(left=df, right=server_df, left_on='eid', right_on='eid', suffixes=('_user', '_server'))
        df = pd.merge(left=df, right=load_df, left_on=['eid', 'timestamp'], right_on=['eid', 'timestamp'])
        df = pd.merge(left=df, right=service_df, left_on='sid', right_on='sid', suffixes=('_server', '_service'))

        t_df = pd.merge(left=load_df, right=server_df, left_on='eid', right_on='eid').reset_index(drop=True)
        t_df = t_df[['timestamp', 'eid', 'computing', 'storage', 'bandwidth']]
        service_df = pd.read_csv(get_path('service.csv'))
        temp_df = pd.merge(inv_df, service_df, left_on='sid', right_on='sid').reset_index(drop=True)
        t_df = pd.merge(t_df, temp_df, left_on=['timestamp', 'eid'], right_on=['timestamp', 'eid'], how='left',
                        suffixes=('_server', '_service')).reset_index(drop=True)
        t_df = t_df[['timestamp', 'eid', 'computing_server', 'storage_server', 'bandwidth_server', 'computing_service',
                     'storage_service', 'bandwidth_service']]
        t_df = t_df.groupby(['timestamp', 'eid', 'computing_server', 'storage_server', 'bandwidth_server']).agg(
            {'computing_service': 'sum', 'storage_service': 'sum', 'bandwidth_service': 'sum'}).sort_values(
            by='timestamp').reset_index()

        d = dict()

        ts = t_df['timestamp'].values
        e_ids = t_df['eid'].values
        tot_c = t_df['computing_service'].values.astype(np.float32)
        tot_s = t_df['storage_service'].values.astype(np.float32)
        tot_b = t_df['bandwidth_service'].values.astype(np.float32)
        self.info = df[['uid', 'eid', 'sid', 'timestamp']].values
        self.info = torch.from_numpy(self.info)

        for i in range(len(ts)):
            d[(e_ids[i], ts[i])] = (tot_c[i], tot_s[i], tot_b[i])

        self.u_lat = torch.from_numpy(df['lat_user'].values.astype(np.float32)).view(-1, 1)
        self.u_lon = torch.from_numpy(df['lon_user'].values.astype(np.float32)).view(-1, 1)
        self.u_speed = torch.from_numpy(df['speed'].values.astype(np.float32)).view(-1, 1)
        self.u_direction = torch.from_numpy(df['direction'].values.astype(np.float32)).view(-1, 1)
        self.e_lat = torch.from_numpy(df['lat_server'].values.astype(np.float32)).view(-1, 1)
        self.e_lon = torch.from_numpy(df['lon_server'].values.astype(np.float32)).view(-1, 1)
        self.e_radius = torch.from_numpy(df['radius'].values.astype(np.float32)).view(-1, 1)
        self.e_c = torch.from_numpy(df['computing_server'].values.astype(np.float32)).view(-1, 1)
        self.e_s = torch.from_numpy(df['storage_server'].values.astype(np.float32)).view(-1, 1)
        self.e_b = torch.from_numpy(df['bandwidth_server'].values.astype(np.float32)).view(-1, 1)
        self.rate_c = torch.from_numpy(df['computing_load'].values.astype(np.float32)).view(-1, 1)
        self.rate_s = torch.from_numpy(df['storage_load'].values.astype(np.float32)).view(-1, 1)
        self.rate_b = torch.from_numpy(df['bandwidth_load'].values.astype(np.float32)).view(-1, 1)
        self.s_c = torch.from_numpy(df['computing_service'].values.astype(np.float32)).view(-1, 1)
        self.s_s = torch.from_numpy(df['storage_service'].values.astype(np.float32)).view(-1, 1)
        self.s_b = torch.from_numpy(df['bandwidth_service'].values.astype(np.float32)).view(-1, 1)
        self.rt = torch.from_numpy(df['rt'].values.astype(np.float32)).view(-1, 1)
        eids = df['eid'].tolist()
        ts = df['timestamp'].tolist()
        self.tot_c = []
        self.tot_s = []
        self.tot_b = []

        for i in range(len(self.u_lat)):
            self.tot_c.append(d[(eids[i], ts[i])][0])
            self.tot_s.append(d[(eids[i], ts[i])][1])
            self.tot_b.append(d[(eids[i], ts[i])][2])

        self.tot_c = torch.from_numpy(np.array(self.tot_c)).view(-1, 1)
        self.tot_b = torch.from_numpy(np.array(self.tot_b)).view(-1, 1)
        self.tot_s = torch.from_numpy(np.array(self.tot_s)).view(-1, 1)

    def __getitem__(self, idx):
        return (
            self.u_lat[idx], self.u_lon[idx], self.u_speed[idx], self.u_direction[idx], self.e_lat[idx],
            self.e_lon[idx],
            self.e_radius[idx], self.e_c[idx], self.e_s[idx], self.e_b[idx], self.rate_c[idx], self.rate_s[idx],
            self.rate_b[idx], self.s_c[idx], self.s_s[idx], self.s_b[idx], self.tot_c[idx], self.tot_s[idx],
            self.tot_b[idx],
            self.rt[idx])

    def __len__(self):
        return len(self.u_lat)


class NutNet(nn.Module):

    def __init__(self):
        super(NutNet, self).__init__()

        self.qos_net = nn.Sequential(
            PreLayer(19, [64, 32, 16, 8, 4, 1])
        )

    def forward(self, u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s, rate_b,
                s_c, s_s, s_b, tot_c, tot_s, tot_b):
        x = torch.concat(
            (u_lat, u_lon, u_speed, u_direction, e_lat, e_lon, e_radius, e_c, e_s, e_b, rate_c, rate_s, rate_b,
             s_c, s_s, s_b, tot_c, tot_s, tot_b), -1)
        x = self.qos_net(x)
        return x


class Best:
    def __init__(self,
                 user_df,
                 server_df,
                 load_df,
                 service_df,
                 inv_df,
                 k):
        super(Best, self).__init__()
        self.dataset = MyDataset(user_df, server_df, load_df, service_df, inv_df, k, 1)

        self.net = NutNet()
        # self.edge_index = torch.LongTensor(generate_graph(server_df, 3, 600))
        self.edge_index = torch.LongTensor(pd.read_csv(get_path('edges.csv')).values.T)

    def get_dataloaders(self, scope, split):
        data_size = len(self.dataset)
        train_valid_size = int(scope * data_size)
        test_size = data_size - train_valid_size

        train_size = int(split * train_valid_size)
        valid_size = train_valid_size - train_size

        train_dataset, valid_dataset, test_dataset = random_split(self.dataset, [train_size, valid_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=256, num_workers=4,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=256, num_workers=4,
                                  shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, num_workers=4,
                                 shuffle=False)

        return train_loader, valid_loader, test_loader
