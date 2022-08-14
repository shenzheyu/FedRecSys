import re

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

from util.user_cluster_data import RatingSample
from util.user_cluster_data import RetrieveSample


class MovielensDataset(Dataset):
    genres_list = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
                   'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                   'Thriller', 'War', 'Western']
    age_list = [1, 18, 25, 35, 45, 50, 56]
    feature_list = ['user_id', 'user_gender', 'user_age', 'user_occupation', 'user_zip_code', 'movie_id', 'movie_title',
                    'movie_year', 'movie_genres', 'rating_timestamp']
    label_list = ['rating', 'retrieve']

    def __init__(self, rating_samples: list[RatingSample], retrieve_samples: list[RetrieveSample]):
        super().__init__()
        self.features_list = []
        self.labels_list = []
        for rating_sample in rating_samples:
            features = {'user_id': rating_sample.user_id, 'user_gender': 1 if rating_sample.gender == 'F' else 0,
                        'user_age': OneHotEncoder().fit(np.array(self.age_list).reshape(-1, 1)).transform(
                            [[rating_sample.age]]).toarray().reshape(-1),
                        'user_occupation': OneHotEncoder().fit(np.arange(21).reshape(-1, 1)).transform(
                            [[rating_sample.occupation]]).toarray().reshape(-1),
                        'user_zipcode': rating_sample.zip_code, 'movie_id': rating_sample.movie_id,
                        'movie_title': (self.title_encode(rating_sample.title))[0],
                        'movie_year': int((self.title_encode(rating_sample.title))[1]),
                        'movie_genres': MultiLabelBinarizer().fit([self.genres_list]).transform(
                            [{g for g in rating_sample.genres.split('|')}]).reshape(-1),
                        'rating_timestamp': rating_sample.timestamp}

            labels = {'rating': rating_sample.rating}

            self.features_list.append(features)
            self.labels_list.append(labels)
        for retrieve_sample in retrieve_samples:
            features = {'user_id': retrieve_sample.user_id, 'user_gender': 1 if retrieve_sample.gender == 'F' else 0,
                        'user_age': OneHotEncoder().fit(np.array(self.age_list).reshape(-1, 1)).transform(
                            [[retrieve_sample.age]]).toarray().reshape(-1),
                        'user_occupation': OneHotEncoder().fit(np.arange(21).reshape(-1, 1)).transform(
                            [[retrieve_sample.occupation]]).toarray().reshape(-1),
                        'user_zipcode': retrieve_sample.zip_code, 'movie_id': retrieve_sample.movie_id,
                        'movie_title': (self.title_encode(retrieve_sample.title))[0],
                        'movie_year': int((self.title_encode(retrieve_sample.title))[1]),
                        'movie_genres': MultiLabelBinarizer().fit([self.genres_list]).transform(
                            [{g for g in retrieve_sample.genres.split('|')}]).reshape(-1),
                        'rating_timestamp': retrieve_sample.timestamp}

            labels = {'retrieve': 1 if retrieve_sample.watch else 0}

            self.features_list.append(features)
            self.labels_list.append(labels)

    def __getitem__(self, item):
        return self.features_list[item], self.labels_list[item]

    def __len__(self):
        return len(self.features_list)

    def title_encode(self, string):
        pattern = re.compile('(.*)\((\d+)\)')
        m = pattern.match(string)
        title_without_year = m.group(1)
        year = m.group(2)

        return title_without_year.strip().split(' '), year
