import tensorflow as tf
import pandas as pd


def read_bx_csv(path):
    return pd.read_csv(path, encoding='ISO-8859-1', delimiter=';', doublequote=False, escapechar='\\')


def train_test_split(ds_or_df, train_fraction=0.8):
    if not isinstance(ds_or_df, tf.data.Dataset):
        assert isinstance(ds_or_df, pd.DataFrame)
        ds = tf.data.Dataset.from_tensor_slices(dict(ds_or_df))
    else:
        ds = ds_or_df

    train_size = int(len(ds) * train_fraction)
    test_size = len(ds) - train_size

    ds = ds.shuffle(len(ds), seed=42, reshuffle_each_iteration=False)

    train = ds.take(train_size)
    test = ds.skip(train_size).take(test_size)
    return train, test


def filter_ratings(ratings, book_ratings_min_count=20, user_ratings_min_count=5, user_ratings_max_count=0):
    """ "Next, we also removed book titles with fewer than 20 overall
    mentions. Only community members with at least 5 ratings
    each were kept.
    - this way the resulting books may not have 20 ratings (since some users will be removed,
     but the ratings are from users with at least 5 kept ratings (of kept books)"
        -- the paper
    """

    # remove books with less than 20 ratings
    book_ratings_count = ratings['ISBN'].value_counts().rename('book_ratings_count')
    ratings = ratings.merge(book_ratings_count, on='ISBN')
    ratings = ratings[ratings.book_ratings_count >= book_ratings_min_count]

    # remove users with less than 5 ratings
    user_ratings_count = ratings['User-ID'].value_counts().rename('user_ratings_count')
    ratings = ratings.merge(user_ratings_count, on='User-ID')
    ratings = ratings[ratings.user_ratings_count >= user_ratings_min_count]

    # remove users with more than 200 ratings
    if user_ratings_max_count > 0:
        ratings = ratings[ratings.user_ratings_count <= user_ratings_max_count]

    return ratings


def squash_user_ids(ratings):
    """ Since after ds filtering we have less users, we should transform user ids to be continuous """
    unique_users = new_to_orig_user_id = ratings['User-ID'].unique()
    orig_to_new_user_id = dict(zip(unique_users, range(len(unique_users))))
    # %%
    ratings['User-ID'] = ratings['User-ID'].map(orig_to_new_user_id)

    return new_to_orig_user_id, orig_to_new_user_id