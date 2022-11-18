import numpy as np
import pandas as pd
import torch


class MovieLensDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, groups_num=0, task_num=1):
        data_frame = pd.read_csv(dataset_path)
        user_ids = data_frame.user_id.unique().tolist()
        self.user_groups = self.get_user_groups(data_frame, user_ids, groups_num)
        data = data_frame.to_numpy()[:, 0:]
        self.categorical_data = data[:, 0: 24].astype(np.int)
        # self.categorical_data = np.concatenate((data[:, 0: 2].astype(np.int), data[:, 6: 24].astype(np.int)), axis=1)
        self.numerical_data = data[:, 24: -2].astype(np.float32)
        if task_num == 1:
            self.labels = np.expand_dims(data[:, -1].astype(np.float32), axis=1)
        elif task_num == 2:
            self.labels = data[:, -2:].astype(np.float32)
        else:
            raise ValueError(f"Unknown task num {task_num}")
        self.numerical_num = self.numerical_data.shape[1]
        # self.field_dims = np.max(self.categorical_data, axis=0) + 1
        self.field_dims = np.array([6041, 3953, 2, 7, 21, 100, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        # self.field_dims = np.array([6041, 3953, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]

    def update_categorical_data(self, index, value):
        self.categorical_data[index] = value

    def get_user_groups(self, data_frame, user_ids, groups_num=0):
        user_groups = {}
        for user_id in user_ids:
            user_groups[user_id] = []
        for index, row in data_frame.iterrows():
            user_groups[row['user_id']].append(index)

        if groups_num > 0:
            merged_groups = {}
            for idx in range(groups_num):
                merged_groups[idx] = []
            for user_id, idxs in user_groups.items():
                merged_groups[user_id % groups_num].extend(idxs)
        else:
            merged_groups = user_groups
        return merged_groups


class MovieLensDatasetWithBehavior(MovieLensDataset):
    def __init__(self, dataset_path, groups_num=0, task_num=1):
        super().__init__(dataset_path, groups_num, task_num)

        self.user_behavior_dict = {}

        self.query_item = np.zeros((self.labels.shape[0]), dtype=np.long)
        self.user_behavior = np.zeros((self.labels.shape[0], 1), dtype=np.long)
        self.user_behavior_length = np.zeros((self.labels.shape[0]), dtype=np.long)

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.query_item[index], \
               self.user_behavior[index], self.user_behavior_length[index], self.labels[index]

    def get_user_behaviors(self, user_behavior_dict=None):
        if user_behavior_dict is None:
            self.user_behavior_dict = {}
            for sample_idx in range(self.categorical_data.shape[0]):
                user_id = self.categorical_data[sample_idx, 0]
                movie_id = self.categorical_data[sample_idx, 1]
                if user_id not in self.user_behavior_dict.keys():
                    self.user_behavior_dict[user_id] = []
                self.user_behavior_dict[user_id].append(movie_id)
        else:
            self.user_behavior_dict = user_behavior_dict

    def update_behavior_data(self):
        seq_len = max([len(behavior) for behavior in list(self.user_behavior_dict.values())])
        self.query_item = np.zeros((self.labels.shape[0]), dtype=np.long)
        self.user_behavior = np.zeros((self.labels.shape[0], seq_len), dtype=np.long)
        self.user_behavior_length = np.zeros((self.labels.shape[0]), dtype=np.long)
        for sample_idx in range(self.categorical_data.shape[0]):
            user_id = self.categorical_data[sample_idx, 0]
            movie_id = self.categorical_data[sample_idx, 1]
            self.query_item[sample_idx] = movie_id
            if user_id in self.user_behavior_dict.keys():
                for behavior_idx, behavior in enumerate(self.user_behavior_dict[user_id]):
                    self.user_behavior[sample_idx, behavior_idx] = behavior
                self.user_behavior_length[sample_idx] = len(self.user_behavior_dict[user_id])
            else:
                self.user_behavior_length[sample_idx] = 0

    def add_offset(self):
        global_offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.long)
        self.categorical_data = self.categorical_data + np.expand_dims(global_offsets, axis=0)
        self.query_item = self.query_item + self.field_dims[0]
        self.user_behavior = self.user_behavior + self.field_dims[0]
