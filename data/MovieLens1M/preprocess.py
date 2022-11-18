import argparse
import re

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--split_train_test', type=bool, default=False)
parser.add_argument('--client_num', type=int, default=0)
args = parser.parse_args()

user_column_names = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
movie_column_names = ['movie_id', 'title', 'genres']
rating_column_names = ['user_id', 'movie_id', 'rating', 'timestamp']

df_users = pd.read_csv('./users.dat', delimiter='::', header=None, names=user_column_names, engine='python')
df_movies = pd.read_csv('./movies.dat', delimiter='::', header=None, names=movie_column_names,
                        encoding='latin-1', engine='python')
df_items = pd.read_csv('./ratings.dat', delimiter='::', header=None, names=rating_column_names, engine='python')

df_data = pd.merge(pd.merge(df_users, df_items, left_on='user_id', right_on='user_id', how='inner'), df_movies,
                   left_on='movie_id', right_on='movie_id', how='inner')

df_data['gender'] = df_data['gender'].astype('category')
df_data['age'] = df_data['age'].astype('category')
df_data['zip_code'] = df_data['zip_code'].map(lambda x: int(x[:2])).astype('category')
cat_columns = df_data.select_dtypes(['category']).columns
df_data[cat_columns] = df_data[cat_columns].apply(lambda x: x.cat.codes)

timestamp_max = df_data['timestamp'].max()
timestamp_min = df_data['timestamp'].min()
df_data['timestamp'] = df_data['timestamp'].map(lambda x: (x - timestamp_min) / (timestamp_max - timestamp_min))

pattern = re.compile('(.*)\((\d+)\)')
df_data['year'] = df_data['title'].map(lambda x: int(pattern.match(x).group(2)))
year_max = df_data['year'].max()
year_min = df_data['year'].min()
df_data['year'] = df_data['year'].map(lambda x: (x - year_min) / (year_max - year_min))

genre_iter = (set(x.split('|')) for x in df_data['genres'])
genres = sorted(set.union(*genre_iter))
dummies = pd.DataFrame(np.zeros((len(df_data), len(genres)), dtype=np.int8), columns=genres)
for i, gen in enumerate(df_data['genres']):
    dummies.loc[i, gen.split('|')] = 1
df_data = df_data.join(dummies.add_prefix('genres_'))

category_cols = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip_code']
category_cols.extend(['genres_' + x for x in genres])
numerical_cols = ['timestamp', 'year']
label_cols = ['rating', 'click']

click_list = []
for rating in df_data['rating']:
    if rating == 5:
        click_list.append([1])
    else:
        click_list.append([0])
df_click = pd.DataFrame(np.array(click_list), columns=['click'])
df_data = pd.concat([df_data, df_click], axis=1)

df_category_data = df_data[category_cols]
df_numerical_data = df_data[numerical_cols]
df_label_data = df_data[label_cols]

# user_id,movie_id,
# categorical_2 (gender),
# categorical_3 (age),
# categorical_4 (occupation),
# categorical_5 (zip_code),
# categorical_6,categorical_7,categorical_8,categorical_9,categorical_10,categorical_11,categorical_12,categorical_13,categorical_14,categorical_15,categorical_16,categorical_17,categorical_18,categorical_19,categorical_20,categorical_21,categorical_22,categorical_23
# numerical_1 (timestamp)
# numerical_2 (year)
# rating (label)
new_category_cols = ['user_id'] + ['movie_id'] + ['categorical_{}'.format(i) for i in range(2, len(category_cols))]
df_category_data.columns = new_category_cols
new_numerical_cols = ['numerical_{}'.format(i) for i in range(1, len(numerical_cols) + 1)]
df_numerical_data.columns = new_numerical_cols

df_final_data = pd.concat([df_category_data, df_numerical_data, df_label_data], axis=1)
print('categorical col num: {}; numerical col num: {}'.format(len(category_cols) - 2, len(numerical_cols)))

df_final_data = df_final_data.sample(frac=1.0)
if args.split_train_test:
    df_train_data = df_final_data.head(int(df_final_data.shape[0] * 0.8))
    df_test_data = df_final_data.tail(int(df_final_data.shape[0] * 0.2))
    df_train_data.to_csv('./train.csv', index=False)
    df_test_data.to_csv('./test.csv', index=False)
    if args.client_num > 0:
        for client_idx in range(args.client_num):
            df_train_data.query(f'user_id % {args.client_num} == {client_idx}').to_csv(f'./train_{client_idx}.csv',
                                                                                       index=False)
else:
    df_final_data.to_csv('./data.csv', index=False)
