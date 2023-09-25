#%%
from pathlib import Path
from dataset import read_bx_csv

data_root = Path('data/BX-CSV-Dump')
users = read_bx_csv(data_root / 'BX-Users.csv')
books = read_bx_csv(data_root / 'BX-Books.csv')
ratings = read_bx_csv(data_root / 'BX-Book-Ratings.csv')

# create wide form of ratings
ratings_wide = ratings.merge(books, on='ISBN').merge(users, on='User-ID').drop(columns=[f'Image-URL-{size}' for size in 'SML'])
ratings_wide.head()
#%%
from dataset import filter_ratings
f_ratings_wide = filter_ratings(ratings_wide, user_ratings_max_count=200)
#%%
# clip user age to 100
f_ratings_wide['Age'] = f_ratings_wide['Age'].clip(upper=100)
#%%
# aggregate users from filtered ratings
f_users = f_ratings_wide.groupby('User-ID').first().reset_index()
f_users.Age.hist(bins=20)
#%%
# replace n/a in user age with -100 (for bucketing to onehot - unknown age will get its own category)
f_ratings_wide['Age'] = f_ratings_wide['Age'].fillna(-100)
#%%
# aggregate books from filtered ratings
f_books = f_ratings_wide.groupby('ISBN').first().reset_index()
f_books[f_books['Year-Of-Publication'] > 0]['Year-Of-Publication'].hist(bins=20)
#%%
f_ratings_wide.isna().sum()
#%%
import numpy as np


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from statistics import LinearRegression
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd

from hash_transform import HashingTransformer
#%%

# fix Location
#   that really could be important, esp. if people cannot read in all languages in the dataset
#  DONE in rating_nn.ipynb
categoricals = ['Book-Author', 'Publisher'] #'Location']

#%%
from sklearn.svm import SVR

preprocessor = ColumnTransformer(transformers=[
    ('bucketize_yop', KBinsDiscretizer(n_bins=12, strategy='kmeans'), ['Year-Of-Publication']),
    ('bucketize_age', KBinsDiscretizer(n_bins=10), ['Age']),
    ('onehot', OneHotEncoder(), categoricals + ['User-ID', 'ISBN']),
    # ('hash', HashingTransformer(bit_size=128), ['User-ID', 'ISBN']),
], remainder='drop')

X = preprocessor.fit_transform(f_ratings_wide)
y = f_ratings_wide['Book-Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svr = SVR(epsilon=1, verbose=2)
svr.fit(X_train, y_train)

# evaluate (R^2 and RMSE)
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
# square root of mse
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}', f'MSE: {mse}', sep='\n')
# R^2
print(f'R^2: {svr.score(X_test, y_test)}')


# save estimator using pickle
import pickle
with open('svr.pickle', 'wb') as f:
    pickle.dump(svr, f)


#%%
