from typing import Dict

import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_dataset(ratings, minmax_columns=('Year-Of-Publication', 'Age'), std_columns=('book_ratings_count', 'user_ratings_count')):
    # make a copy of the dataset, so that the original is not modified
    ratings = ratings.copy()

    minmax_columns = list(minmax_columns)  # so that pandas indexing works (tuple did not)
    std_columns = list(std_columns)

    # apply min/max scaler to numeric columns
    # strictly, this should be done only on the training set, but for simplicity we do it on the whole dataset
    ratings[minmax_columns] = MinMaxScaler().fit_transform(ratings[minmax_columns])

    # TODO add e.g. 'book_rating_mean', 'user_rating_mean'?
    ratings[std_columns] = StandardScaler().fit_transform(ratings[std_columns])
    return ratings


embedding_dimensions = {
    'User-ID': 32,
    'Book-Title': 32,
    'City': 10,
    'State': 7,
    'Country': 5,
}


class RankingModel(tf.keras.Model):
    def __init__(self, dataset, embedding_dimensions: Dict[str, int], l2_regularization=0.0, dropout=0.0):
        super().__init__()

        self.embeddings = {
            'User-ID': tf.keras.layers.Embedding(dataset['User-ID'].nunique() + 1, embedding_dimensions.pop('User-ID'))
        }

        # Create embedding layers for columns (Book-Title, City, State, Country).
        for field, embedding_dimension in embedding_dimensions.items():
            vocab = dataset[field].unique()
            self.embeddings[field] = tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=vocab),
                tf.keras.layers.Embedding(len(vocab) + 1, embedding_dimension)
            ])

        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)),
            tf.keras.layers.Dropout(dropout),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)  # alternatively, sigmoid could be used if containing predictions in a range would be needed
        ])

    def call(self, inputs):
        for field in inputs:
            value = inputs[field]

            if field in self.embeddings:
                # embed the (string) field
                inputs[field] = self.embeddings[field](value)
            else:
                # expand the dimension of the numeric field
                inputs[field] = tf.expand_dims(value, -1)

        # Concatenate embeddings and other features and feed into rating model.
        ratings = self.ratings(tf.concat(list(inputs.values()), axis=1))

        # convert to 1-10 rating (but continuous and may exceed the bounds)
        ratings = (ratings + 1) * 4.5 + 1

        return ratings


class BXRankingModel(tfrs.models.Model):
  def __init__(self, ranking_model: tf.keras.Model):
    super().__init__()
    self.ranking_model = ranking_model
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(features)

  def compute_loss(self, features: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop('Book-Rating')

    rating_predictions = self(features)

    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)


def factorized_top_k(model, ks, user_ids, ratings_df, test_ratings_df):


    # Generate some example data
    num_users = 100
    num_books = 500
    num_test_ratings = 1000
    batch_size = 50
    top_ks = [1, 5, 10]  # Different levels of top-K to evaluate

    # Training data
    user_ids = np.random.choice(num_users, 1000)
    book_ids = np.random.choice(num_books, 1000)
    ratings = np.random.rand(1000)

    # Test data
    test_user_ids = test_ratings_df['User-ID'].to_numpy()
    test_book_ids = test_ratings_df['ISBN'].to_numpy()  # fast enough string id?
    test_ratings = test_ratings_df['Book-Rating'].to_numpy()

    # Initialize variables to keep track of top-K accuracy
    correct_counts = {k: 0 for k in top_ks}
    total_count = 0

    # Evaluate top-K for test set in batches
    for i in range(0, num_test_ratings, batch_size):
        batch_user_ids = test_user_ids[i:i + batch_size]
        batch_book_ids = test_book_ids[i:i + batch_size]

        # Generate all possible user-book pairs for this batch of users
        user_vector = np.tile(batch_user_ids, num_books)
        book_vector = np.repeat(np.arange(num_books), len(batch_user_ids))

        X_test = np.column_stack([user_vector, book_vector])

        # Get predicted ratings for all user-book pairs in batch
        predicted_ratings = model.predict(X_test)

        # Reshape to have shape (num_users_in_batch, num_books)
        predicted_ratings_matrix = predicted_ratings.reshape(-1, num_books)

        # Use tf.math.in_top_k to find if the true best book is in top K
        for user_idx, true_best_book in enumerate(batch_book_ids):
            for k in top_ks:
                correct = tf.math.in_top_k(true_best_book, predicted_ratings_matrix[user_idx], k)
                correct_counts[k] += tf.cast(correct, dtype=tf.int32).numpy()

        total_count += len(batch_user_ids)

    # Compute top-K accuracy
    for k in top_ks:
        top_k_accuracy = correct_counts[k] / total_count
        print(f"Top-{k} accuracy: {top_k_accuracy}")



