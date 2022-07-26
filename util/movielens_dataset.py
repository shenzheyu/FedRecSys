from torch.utils.data import Dataset
from util.user_cluster_data import RatingSample
from util.user_cluster_data import RetrieveSample
import torch


class MovielensDataset(Dataset):
    def __init__(self, rating_samples: list[RatingSample], retrieve_samples: list[RetrieveSample]):
        super().__init__()
        self.features = []
        self.labels = []
        self.tasks = []
        for rating_sample in rating_samples:
            features = ['user_id/' + str(rating_sample.user_id), 'gender/' + rating_sample.gender,
                       'age/' + str(rating_sample.age), 'occupation/' + str(rating_sample.occupation),
                       'zip_code/' + rating_sample.zip_code, 'movie_id/' + str(rating_sample.movie_id),
                       'title/' + rating_sample.title, 'genres/' + rating_sample.genres,
                       'timestamp/' + str(rating_sample.timestamp)]
            label = [rating_sample.rating, 0]
            task = [1, 0]
            self.features.append([hash(feature) for feature in features])
            self.labels.append(label)
            self.tasks.append(task)
        for retrieve_sample in retrieve_samples:
            feature = ['user_id/' + str(retrieve_sample.user_id), 'gender/' + retrieve_sample.gender,
                       'age/' + str(retrieve_sample.age), 'occupation/' + str(retrieve_sample.occupation),
                       'zip_code/' + retrieve_sample.zip_code, 'movie_id/' + str(retrieve_sample.movie_id),
                       'title/' + retrieve_sample.title, 'genres/' + retrieve_sample.genres,
                       'timestamp/' + str(retrieve_sample.timestamp)]
            if retrieve_sample.watch:
                label = [0, 1]
            else:
                label = [0, 0]
            task = [0, 1]
            self.features.append([hash(feature) for feature in features])
            self.labels.append(label)
            self.tasks.append(task)

    def __getitem__(self, item):
        return torch.tensor(self.features[item]), torch.tensor(self.labels[item]), torch.tensor(self.tasks[item])

    def __len__(self):
        return len(self.features)
