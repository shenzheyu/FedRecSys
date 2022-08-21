import pandas as pd
from torch.utils.data import Dataset

from model.embedding_lookup import EmbeddingLookup
from util.movielens_dataset import MovielensDataset
from util.user_cluster_data import RatingSample
from util.user_cluster_data import RetrieveSample
from util.user_cluster_data import UserClusterData


def data_prepare() \
        -> (Dataset, Dataset, Dataset, list[list[float]], dict[int, int]):
    user_column_names = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    movie_column_names = ['movie_id', 'title', 'genres']
    rating_column_names = ['user_id', 'movie_id', 'rating', 'timestamp']

    users_df = pd.read_csv('data/MovieLens1M/users.dat', delimiter='::', header=None, names=user_column_names, engine='python')
    movies_df = pd.read_csv('data/MovieLens1M/movies.dat', delimiter='::', header=None, names=movie_column_names,
                            encoding='latin-1', engine='python')
    ratings_df = pd.read_csv('data/MovieLens1M/ratings.dat', delimiter='::', header=None, names=rating_column_names,
                             engine='python')

    print(f'load {users_df.shape[0]} users from data/users.dat')
    print(f'load {movies_df.shape[0]} movies from data/movies.dat')
    print(f'load {ratings_df.shape[0]} ratings from data/ratings.dat')

    user_cluster_datas = {}
    for index, rating_series in ratings_df.iterrows():
        # if index > 10000:
        #     break

        if index % 10000 == 0:
            print(f'process {index} rating row')

        user_id = rating_series.user_id
        movie_id = rating_series.movie_id

        user_series = users_df.loc[users_df['user_id'] == user_id].iloc[0]
        movie_series = movies_df.loc[movies_df['movie_id'] == movie_id].iloc[0]

        rating_sample = RatingSample()
        rating_sample.add_user_id(user_id)
        rating_sample.add_user_info(user_series.gender, user_series.age, user_series.occupation, user_series.zip_code)
        rating_sample.add_movie_info(movie_series.movie_id, movie_series.title, movie_series.genres)
        rating_sample.add_rating_info(rating_series.rating, rating_series.timestamp)

        retrieve_sample = RetrieveSample()
        retrieve_sample.add_user_id(user_id)
        retrieve_sample.add_user_info(user_series.gender, user_series.age, user_series.occupation, user_series.zip_code)
        retrieve_sample.add_movie_info(movie_series.movie_id, movie_series.title, movie_series.genres)
        retrieve_sample.add_retrieve_info(True, rating_series.timestamp)

        if user_id in user_cluster_datas.keys():
            user_cluster_data = user_cluster_datas[user_id]
        else:
            user_cluster_data = UserClusterData()
            user_cluster_datas[user_id] = user_cluster_data
        user_cluster_data.add_rating_sample(rating_sample)
        user_cluster_data.add_positive_retrieve_sample(retrieve_sample)
    print(f'processed {len(user_cluster_datas)} users\' data')

    for user_cluster_data in user_cluster_datas.values():
        user_cluster_data.generate_retrieve_negative_sample(movies_df)
    print('generate negative sample for retrieve data')

    return user_cluster_datas


def get_dataset(user_cluster_datas: {int: UserClusterData}):
    train_rating_samples = []
    train_retrieve_samples = []
    val_rating_samples = []
    val_retrieve_samples = []
    test_rating_samples = []
    test_retrieve_samples = []
    for user_cluster_data in user_cluster_datas.values():
        user_train_rating_samples, user_val_rating_samples, user_test_rating_samples = \
            user_cluster_data.get_rating_samples([6, 2, 2])
        user_train_retrieve_samples, user_val_retrieve_samples, user_test_retrieve_samples = \
            user_cluster_data.get_retrieve_samples([6, 2, 2])
        train_rating_samples += user_train_rating_samples
        train_retrieve_samples += user_train_retrieve_samples
        val_rating_samples += user_val_rating_samples
        val_retrieve_samples += user_val_retrieve_samples
        test_rating_samples += user_test_rating_samples
        test_retrieve_samples += user_test_retrieve_samples
    print('merge samples for train, val and test')

    train_dataset = MovielensDataset(train_rating_samples, train_retrieve_samples)
    val_dataset = MovielensDataset(val_rating_samples, val_retrieve_samples)
    test_dataset = MovielensDataset(test_rating_samples, test_retrieve_samples)

    print('generate dataset for train, val and test')
    return train_dataset, val_dataset, test_dataset
