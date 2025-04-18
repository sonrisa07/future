from tqdm import tqdm

from utils import get_path
import pandas as pd
import numpy as np


def getSvc():
    service_df = pd.read_csv(get_path('service.csv'))
    inv_df = pd.read_csv(get_path('invocation.csv'))
    df = pd.merge(left=inv_df, right=service_df, left_on='sid', right_on='sid')
    print(df.columns)
    g = df.groupby(by=['eid', 'timestamp'], as_index=False).agg({
        'computing': 'sum',
        'storage': 'sum',
        'bandwidth': 'sum'
    })

    d = dict()
    for row in g.itertuples(index=False):
        eid = row.eid
        timestamp = row.timestamp
        computing_sum = row.computing
        storage_sum = row.storage
        bandwidth_sum = row.bandwidth
        d[(eid, timestamp)] = np.array([computing_sum, storage_sum, bandwidth_sum])

    return d
