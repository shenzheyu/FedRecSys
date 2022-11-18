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

    def __init__(self, dataset_path, groups_num=0, task_num=2):
        data_frame = pd.read_csv(dataset_path)
        user_ids = data_frame.search_id.unique().tolist()
        self.user_groups = self.get_user_groups(data_frame, user_ids, groups_num)
        data = data_frame.to_numpy()[:, 0:]
        self.categorical_data = data[:, :17].astype(np.int)
        self.numerical_data = data[:, 17: -2].astype(np.float32)
        if task_num == 2:
            self.labels = data[:, -2:].astype(np.float32)
        elif task_num == 1:
            self.labels = data[:, -2:-1].astype(np.float32)
        else:
            raise ValueError(f"Unknown task num {task_num}")
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]

    def get_user_groups(self, data_frame, user_ids, groups_num=0):
        user_groups = {}
        for user_id in user_ids:
            user_groups[user_id] = []
        for index, row in data_frame.iterrows():
            user_groups[row['search_id']].append(index)

        if groups_num > 0:
            merged_groups = {}
            for idx in range(groups_num):
                merged_groups[idx] = []
            for user_id, idxs in user_groups.items():
                merged_groups[user_id % groups_num].extend(idxs)
        else:
            merged_groups = user_groups
        return merged_groups
