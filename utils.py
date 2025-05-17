import os.path

import math
import numpy as np
import pandas as pd
import sklearn.cluster
import torch
from scipy.special import kl_div
from torch.utils.data import Subset

from config import (
    proRoot,
    fluctuation_min,
    fluctuation_max,
    k_rt,
    MAX_DELAY,
    BASE_DELAY,
    cluster_size,
    r,
)


class StandardScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sort_dataset(subset, dataset, i, j):
    indices = subset.indices
    sorted_indices = sorted(indices, key=lambda idx: dataset[idx][i][j].item())
    return Subset(dataset, sorted_indices)


def positional_encoding(x, device):
    k, d = x.shape[-2], x.shape[-1]
    mapping = torch.zeros((k, d), requires_grad=False).to(device)
    pos = torch.arange(
        0, k, device=device, dtype=torch.float32, requires_grad=False
    ).unsqueeze(1)
    col = torch.tensor(
        [10000 ** (2 * i / d) for i in range(0, d, 2)],
        device=device,
        dtype=torch.float32,
        requires_grad=False,
    )
    mapping[:, 0::2] = torch.sin(pos / col)
    mapping[:, 1::2] = torch.cos(pos / col)
    return mapping


def get_path(filename: str, root: str = proRoot) -> str:
    return os.path.join(root, filename)


def is_covered(u_lat, u_lon, s_lat, s_lon, s_radius):
    return haversine(u_lat, u_lon, s_lat, s_lon) <= s_radius


def euclidean_distance(point1: tuple[int, int], point2: tuple[int, int]) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def find_servers(
    lat, lon, e_ids: list, e_lat: list, e_lon: list, e_radius: list
) -> list:
    temp = []
    for i in range(len(e_ids)):
        if is_covered(lat, lon, e_lat[i], e_lon[i], e_radius[i]):
            temp.append(e_ids[i])
    return temp


def pro_fluctuation(x: int) -> int:
    min_t = max(fluctuation_min, 500 - x)
    max_t = min(fluctuation_max, 9900 - x)
    t = r.randint(min_t, max_t)
    return x + t


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径（公里）
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance.item() * 1000  # 返回距离（米）


def haversine_np(lat1, lon1, lat2, lon2: np.ndarray) -> np.ndarray:
    R = 6371  # 地球半径（公里）
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance * 1000  # 返回距离（米）


# 找到用户移动后的位置
def update_user_position(lat, lon, speed, direction, time):
    # 将速度和时间转化为移动的距离（米）
    distance = speed * time

    # 将角度转化为弧度
    direction_rad = math.radians(direction)

    # 计算新的纬度和经度
    delta_lat = distance * math.cos(direction_rad) / 111320  # 每度纬度约111.32公里
    delta_lon = (
        distance * math.sin(direction_rad) / (111320 * math.cos(math.radians(lat)))
    )

    new_lat = lat + delta_lat
    new_lon = lon + delta_lon

    return new_lat, new_lon


def kl_divergence(p, q):
    """
    计算两个概率分布之间的KL散度。

    参数:
    p (array-like): 第一个概率分布。
    q (array-like): 第二个概率分布。

    返回:
    float: KL散度值。
    """
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)

    # 避免除零错误
    # p = np.clip(p, 1e-10, 1)
    # q = np.clip(q, 1e-10, 1)
    #
    # return np.sum(p * np.log(p / q))

    return kl_div(p, q).sum()


def cal_trans_comp_latency(p, q, kl, mean, density):
    """
    计算传输和处理延迟
    p: 服务资源需求分布
    q: 服务器资源供给分布
    """
    return min(
        (k_rt[0] * kl + k_rt[1] * mean + k_rt[2] * density) * BASE_DELAY, MAX_DELAY
    )  # 两个分布差异越大，节点资源供给越少，延迟越大


# 标准化函数
def standardize(value, mean, std):
    return (value - mean) / std


def m_standardize(data: np.ndarray) -> np.ndarray:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# 综合考虑过去 k 个时间片计算方向变化率
def calculate_direction_change_rate(directions, k):
    if k == 1:
        return 0
    changes = [abs(directions[i] - directions[i - 1]) for i in range(1, k)]
    return np.mean(changes).item()


def normalize_to_range(values: np.ndarray, a=0, b=1) -> np.ndarray:
    min_val = np.min(values)
    max_val = np.max(values)

    if max_val - min_val == 0:
        return np.array([a for _ in values])

    normalized_values = (values - min_val) / (max_val - min_val) * (b - a) + a
    return normalized_values


def get_cluster_density(kmeans: sklearn.cluster.KMeans, df: pd.DataFrame):
    clusters = kmeans.labels_
    cluster_density = []
    for i in range(cluster_size):
        tot = np.sum(clusters == i)
        cluster_density.append(tot)
    return np.array(cluster_density) / np.sum(cluster_density)


def convert_percentage_to_decimal(percentage_str):
    percentage_value = float(percentage_str.strip("%"))

    decimal_value = percentage_value / 100

    return decimal_value


def generate_causal_mask_pytorch(k):
    causal_mask = torch.tril(torch.ones((k, k), dtype=torch.int32))
    return causal_mask


def mercator(lat, lon: torch.Tensor) -> torch.Tensor:
    lon = lon * 20037508.342789 / 180
    lat = torch.log(torch.tan((90 + lat) * torch.pi / 360)) / (torch.pi / 180)
    lat = lat * 20037508.342789 / 180
    return torch.stack((lat, lon), dim=-1)


def meters_to_mercator_unit(distance_meters, lat_deg):
    lat_rad = math.radians(lat_deg)
    meters_per_degree = 111320 * math.cos(lat_rad)
    delta_lon_deg = distance_meters / meters_per_degree
    lon1, lat1 = 0.0, lat_deg
    lon2, lat2 = lon1 + delta_lon_deg, lat1

    def wgs84toMercator(lon, lat):
        x = lon * 20037508.342789 / 180
        y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)
        y = y * 20037508.342789 / 180
        return x, y

    x1, y1 = wgs84toMercator(lon1, lat1)
    x2, y2 = wgs84toMercator(lon2, lat2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
