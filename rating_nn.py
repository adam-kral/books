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
