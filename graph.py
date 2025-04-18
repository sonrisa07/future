import pandas as pd
from rich.progress import track

from utils import find_servers, get_path
import numpy as np


def generate_graph(user_df, server_df):
    t_eid = server_df['eid'].values
    t_lat = server_df['lat'].values
    t_lon = server_df['lon'].values
    t_radius = server_df['radius'].values
    edges = [[False for _ in range(len(server_df))] for _ in range(len(server_df))]

    # 将 user_df 转为列表形式，减少访问开销
    user_data = user_df.groupby('uid').apply(lambda g: g.sort_values(by='timestamp')[['lat', 'lon']].values).to_dict()

    for uid, rows in track(user_data.items()):
        pre = []
        for lat, lon in rows:
            eids = find_servers(lat, lon, t_eid, t_lat, t_lon, t_radius)
            for i in range(len(eids)):
                for j in range(i + 1, len(eids)):
                    edges[eids[i]][eids[j]] = True
                    edges[eids[j]][eids[i]] = True
            for x in pre:
                for y in eids:
                    edges[x][y] = True
                    edges[y][x] = True
            pre = eids

    edge_index = []
    for i in range(len(server_df)):
        for j in range(len(server_df)):
            if edges[i][j]:
                edge_index.append([i, j])

    return np.array(edge_index).T


if __name__ == '__main__':
    user_df = pd.read_csv(get_path('user.csv'))
    server_df = pd.read_csv(get_path('server.csv'))
    x = generate_graph(user_df, server_df)
    df = pd.DataFrame(x.T, columns=['source', 'target'])
    df.to_csv(get_path('edges.csv'), index=False)
