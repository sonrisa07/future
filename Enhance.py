import pandas as pd
from utils import get_path

user_df = pd.read_csv(get_path("user.csv"))
srv_df = pd.read_csv(get_path("server.csv"))
load_df = pd.read_csv(get_path("load.csv"))
svc_df = pd.read_csv(get_path("service.csv"))
inv_df = pd.read_csv(get_path("invocation.csv"))

df = pd.merge(
    left=inv_df,
    right=user_df,
    left_on=["uid", "timestamp"],
    right_on=["uid", "timestamp"],
)
df = pd.merge(
    left=df,
    right=srv_df,
    left_on=["eid", "timestamp"],
    right_on=["eid", "timestamp"],
    suffixes=(("_u", "_e")),
)

df.drop(columns=["uid", "sid", "rt"], axis=1, inplace=True)

u_lat = df["lat_u"].values
u_lon = df["lon_u"].values
e_lat = df["lat_e"].values
e_lon = df["lon_e"].values
