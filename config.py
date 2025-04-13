import os
import random

import numpy as np
import torch
import logging
from torch.backends import mps
from rich.logging import RichHandler

base_dir = os.path.dirname(os.path.abspath(__file__))

n_hours = 24
time_slot = 60
root = "./dataset"
proRoot = os.path.join(base_dir, 'CHESTNUT')
user_dir = root + "/user"
service_dir = root + "/service"
server_dir = root + "/server"

act_users = 1

dt_format = "%Y-%m-%d %H:%M:%S"

n_users = 2000
n_servers = 100
cluster_size = 100
n_services = 200
preference = 5
lon_lower, lat_lower = 121.259, 31.05
lon_upper, lat_upper = 121.6400, 31.372

# lon_lower, lat_lower = 0, 0
# lon_upper, lat_upper = 200, 100

max_statistic = 21600
min_tot_timestamp = 239
radius_lower = 1200
radius_upper = 2500

min_percent = 500

fluctuation_min = -8000
fluctuation_max = 1500

static_interval = 3

min_inv = 80
max_inv = 200

SPEED_OF_LIGHT = 3e8  # 光速（米/秒）
BASE_DELAY = 0.8  # 基础延迟（秒）
MAX_DELAY = 1.6
BASE_JITTER = 160  # ms

device = "mps" if mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

columns_users = ["id", "timestamp", "lon", "lat", "speed", "direction"]
columns_servers = ["id", "lon", "lat", "radius", "computing", "storage", "bandwidth"]
columns_invocations = [
    "uid",
    "eid",
    "sid",
    "timestamp",
    "rt",
    "nj",
]
columns_loads = ['timestamp', 'eid', 'computing_load', 'storage_load', 'bandwidth_load']

qos_offsets_range = {"rt": [0, 2], "nj": [0, 2]}

direction_ratio_k = 5

k_rt = [0.5, 0.02, 0.02]

k_distance, k_direction_change_rate, k_bandwidth_ratio = 1, 1, 1

nj_min, nj_max = 0, 5

crash_percent = 0.05
crash_sample = 0.5

logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("rich")

torch.set_default_dtype(torch.float32)

np.random.seed(910)

r = random.Random(91077912)

wsRoot = os.path.join(base_dir, 'ws_dataset')
