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

