import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np

# device onfig
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Train_Data = pd.read_csv("PyTorch/Alex_Motion_Cap_Formatted.csv")

FEATURE_COLUMNS = Train_Data.columns.tolist()[24:]
LABEL_COLUMNS = Train_Data.columns.tolist()[12]

FEATURE_COLUMNS.append(LABEL_COLUMNS)

data = Train_Data[FEATURE_COLUMNS]



print(data.head())

from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Timestamp', inplace=True)

    for i in range(1, n_steps+1):
        df[f'Hand_Top_X(t-{i})'] = df['Hand_Top_X'].shift(i)

    df.dropna(inplace=True)

    return df

lookback = 7
shifted_df = prepare_dataframe_for_lstm(data, lookback)

shifted_df_as_np = shifted_df.to_numpy()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

shifted_df_as_np




