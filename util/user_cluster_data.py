import math

import pandas as pd


class Sample:
    def __init__(self):
        self.user_id = None
        self.gender = None
        self.age = None
        self.occupation = None
        self.zip_code = None
        self.movie_id = None
        self.title = None
        self.genres = None
        self.timestamp = None

    def add_user_id(self, user_id: int):
        self.user_id = user_id

    def add_user_info(self, gender, age, occupation, zip_code):
        self.gender = gender
        self.age = age
        self.occupation = occupation
        self.zip_code = zip_code

    def add_movie_info(self, movie_id, title, genres):
        self.movie_id = movie_id
        self.title = title
        self.genres = genres


class RatingSample(Sample):
    def __init__(self):
        super().__init__()
        self.rating = None

    def add_rating_info(self, rating: int, timestamp: str):
        self.rating = rating
        self.timestamp = timestamp


class RetrieveSample(Sample):
    def __init__(self):
        super().__init__()
        self.watch = None

    def add_retrieve_info(self, watch: bool, timestamp: str):
        self.watch = watch
        self.timestamp = timestamp


class UserClusterData:
    def __init__(self):
        self.rating_samples = []
        self.positive_retrieve_samples = []
        self.negative_retrieve_samples = []
        self.positive_retrieve_samples_num = 0
        self.positive_retrieve_movie_ids = set()
        self.negative_retrieve_movie_ids = set()
        self.negative_retrieve_sample_template = None

    def add_rating_sample(self, rating_sample: RatingSample):
        self.rating_samples.append(rating_sample)

    def add_positive_retrieve_sample(self, retrieve_sample: RetrieveSample):
        self.positive_retrieve_samples_num += 1
        self.positive_retrieve_movie_ids.add(retrieve_sample.movie_id)
        self.positive_retrieve_samples.append(retrieve_sample)

        if self.negative_retrieve_sample_template is None:
            self.negative_retrieve_sample_template = RetrieveSample()
            self.negative_retrieve_sample_template.add_user_id(retrieve_sample.user_id)
            self.negative_retrieve_sample_template.add_user_info(retrieve_sample.gender, retrieve_sample.age,
                                                                 retrieve_sample.occupation, retrieve_sample.zip_code)

    def add_negative_retrieve_sample(self, retrieve_sample: RetrieveSample) -> bool:
        if retrieve_sample.movie_id in self.negative_retrieve_movie_ids:
            return False
        self.negative_retrieve_movie_ids.add(retrieve_sample.movie_id)
        self.negative_retrieve_samples.append(retrieve_sample)
        return True

    def retrieve_negative_sample(self, movies_df: pd.DataFrame) -> bool:
        if self.negative_retrieve_sample_template is None:
            return False

        negative_movies_df = movies_df.loc[~movies_df['movie_id'].isin(self.positive_retrieve_movie_ids)].sample(
            frac=self.positive_retrieve_samples_num, replace=True)
        for index in range(self.positive_retrieve_samples_num):
            movie_series = negative_movies_df.iloc[index]
            negative_retrieve_sample = RetrieveSample()
            negative_retrieve_sample.add_user_id(self.negative_retrieve_sample_template.user_id)
            negative_retrieve_sample.add_user_info(self.negative_retrieve_sample_template.gender,
                                                   self.negative_retrieve_sample_template.age,
                                                   self.negative_retrieve_sample_template.occupation,
                                                   self.negative_retrieve_sample_template.zip_code)
            negative_retrieve_sample.add_movie_info(movie_series.movie_id,
                                                    movie_series.title,
                                                    movie_series.genres)
            negative_retrieve_sample.add_retrieve_info(False, self.positive_retrieve_samples[index].timestamp)
            self.add_negative_retrieve_sample(negative_retrieve_sample)
        return True

    def get_rating_samples(self, portion: list[int]) -> (list[RatingSample], list[RatingSample], list[RatingSample]):
        self.rating_samples.sort(key=lambda sample: sample.timestamp, reverse=False)
        train_sample_num = math.floor(len(self.rating_samples) * portion[0] / sum(portion))
        val_sample_num = math.floor(len(self.rating_samples) * portion[1] / sum(portion))
        return self.rating_samples[0: train_sample_num], \
               self.rating_samples[train_sample_num: train_sample_num + val_sample_num], \
               self.rating_samples[train_sample_num + val_sample_num: -1]

    def get_retrieve_samples(self, portion: list[int]) -> (list[RatingSample], list[RatingSample], list[RatingSample]):
        self.positive_retrieve_samples.sort(key=lambda sample: sample.timestamp, reverse=False)
        self.negative_retrieve_samples.sort(key=lambda sample: sample.timestamp, reverse=False)
        train_sample_num = math.floor(len(self.rating_samples) * portion[0] / sum(portion))
        val_sample_num = math.floor(len(self.rating_samples) * portion[1] / sum(portion))
        return self.positive_retrieve_samples[0: train_sample_num] + \
               self.negative_retrieve_samples[0: train_sample_num], \
               self.positive_retrieve_samples[train_sample_num: train_sample_num + val_sample_num] + \
               self.negative_retrieve_samples[train_sample_num: train_sample_num + val_sample_num], \
               self.positive_retrieve_samples[train_sample_num + val_sample_num: -1] + \
               self.negative_retrieve_samples[train_sample_num + val_sample_num: -1]
