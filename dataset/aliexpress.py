import numpy as np
import pandas as pd
import torch


class AliExpressDataset(torch.utils.data.Dataset):
    """
    AliExpress Dataset
    This is a dataset gathered from real-world traffic logs of the search system in AliExpress
    Reference:
        https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690
        Li, Pengcheng, et al. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. CIKM 2020.
    """

    def __init__(self, dataset_path):
        data_frame = pd.read_csv(dataset_path)
        self.user_num = data_frame.search_id.nunique()
        user_ids = data_frame.search_id.unique().tolist()
        self.user_groups = self.get_user_groups(data_frame, user_ids)
        data = data_frame.to_numpy()[:, 0:]
        self.categorical_data = data[:, :17].astype(np.int)
        self.numerical_data = data[:, 17: -2].astype(np.float32)
        self.labels = data[:, -2:].astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]

    def get_user_groups(self, data_frame, user_ids):
        user_groups = {}
        for user_id in user_ids:
            user_groups[user_id] = []
        for index, row in data_frame.iterrows():
            user_groups[row['search_id']].append(index)
            # series = (data_frame.search_id == user_id)
            # user_groups[user_id] = series[series].index.tolist()
        return user_groups