from turtle import heading
import pandas as pd
import numpy as np
from utils import get_path, haversine, haversine_np
from rich.progress import track

user_df = pd.read_csv(get_path("user.csv"))
srv_df = pd.read_csv(get_path("server.csv"))
load_df = pd.read_csv(get_path("load.csv"))
svc_df = pd.read_csv(get_path("service.csv"))
inv_df = pd.read_csv(get_path("invocation.csv"))

assert isinstance(user_df, pd.DataFrame)
assert isinstance(srv_df, pd.DataFrame)
assert isinstance(load_df, pd.DataFrame)
assert isinstance(svc_df, pd.DataFrame)
assert isinstance(inv_df, pd.DataFrame)

df = pd.merge(
    left=inv_df,
    right=user_df,
    left_on=["uid", "timestamp"],
    right_on=["uid", "timestamp"],
)
df = pd.merge(
    left=df,
    right=srv_df,
    left_on=["eid"],
    right_on=["eid"],
    suffixes=(("_u", "_e")),
)
df.drop(
    columns=["uid", "sid", "rt", "computing", "storage", "bandwidth"],
    axis=1,
    inplace=True,
)

columns = [
    "link_distance",  # 真实距离
    "radial_speed",  # 径向速度
    "tangential_speed",  # 切向速度
    "relative_bearing",  # 方位角
    "normalized_range",  # 归一化距离
    "heading_offset",  # 方向夹角
    "edge_margin",  # 距离边界
]

u_lat = df["lat_u"].to_numpy()
u_lon = df["lon_u"].to_numpy()
e_lat = df["lat_e"].to_numpy()
e_lon = df["lon_e"].to_numpy()
radius = df["radius"].to_numpy()
speed = df["speed"].to_numpy()
direction = df["direction"].to_numpy()

theta = np.mod(np.deg2rad(direction), 2 * np.pi)

# lon0 = np.mean(np.concatenate([u_lon, e_lon]))
# lat0 = np.mean(np.concatenate([u_lat, e_lat]))

lat0, lon0 = 31.229405638455106, 121.44107074878619

fi0 = np.deg2rad(lat0)
R = 6371000

u_x = np.deg2rad(u_lon - lon0) * np.cos(fi0) * R
u_y = np.deg2rad(u_lat - lat0) * R
e_x = np.deg2rad(e_lon - lon0) * np.cos(fi0) * R
e_y = np.deg2rad(e_lat - lat0) * R

dx = u_x - e_x
dy = u_y - e_y

link_dis = np.sqrt(dx**2 + dy**2)
relative_bearing = np.mod(np.arctan2(dx, dy), 2 * np.pi)
radial_speed = speed * np.cos(theta - relative_bearing)
tangential_speed = speed * np.sin(theta - relative_bearing)
normalized_range = link_dis / radius
heading_offset = np.abs(
    np.mod(np.abs(theta - relative_bearing + np.pi), 2 * np.pi) - np.pi
)
edge_margin = radius - link_dis
feature = np.stack(
    [
        link_dis,
        radial_speed,
        tangential_speed,
        relative_bearing,
        normalized_range,
        heading_offset,
        edge_margin,
    ],
    axis=1,
)

pd.DataFrame(feature, columns=columns).to_csv(
    get_path("enhance_invocation.csv"), index=False
)
