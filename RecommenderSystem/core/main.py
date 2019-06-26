import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#Reading users file:

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']

users = pd.read_csv('C:\\Users\\anves\\github\\Kaggle\\RecommenderSystem\\data\\u.user',sep='|', names=u_cols,encoding='latin-1')

# print(users)

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('C:\\Users\\anves\\github\\Kaggle\\RecommenderSystem\\data\\u.data', sep='\t', names=r_cols,encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


items = pd.read_csv('C:\\Users\\anves\\github\\Kaggle\\RecommenderSystem\\data\\u.item', sep='|', names=i_cols,encoding='latin-1')

# print(items.head())

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_train = pd.read_csv('C:\\Users\\anves\\github\\Kaggle\\RecommenderSystem\\data\\ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('C:\\Users\\anves\\github\\Kaggle\\RecommenderSystem\\data\\ua.test', sep='\t', names=r_cols, encoding='latin-1')

# print(ratings_train.head())

# print(ratings_train.shape, ratings_test.shape)

n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

# print(ratings.head())

data_matrix = np.zeros((n_users, n_items))

for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]

from sklearn.metrics.pairwise import pairwise_distances 
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

