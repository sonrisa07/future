from collections import Counter

import networkx as nx
from model.layers import AutoEmbedding
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader, Subset
from torch_geometric_temporal import STConv

from model.STModel import STModel
from module.PreLayer import PreLayer
from utils import StandardScaler, mercator, meters_to_mercator_unit, sort_dataset
from utils import convert_percentage_to_decimal, get_path
import torch.nn as nn
import torch.nn.functional as F


class MyDataset(Dataset):
    def __init__(self, user_df, server_df, load_df, service_df, inv_df, k):
        pass


class Dynamic(nn.Module):
    def __init__(self, user_df, server_df, load_df, service_df, inv_df, k):
        pass

    def get_dataloaders(self, scope, split):
        pass
