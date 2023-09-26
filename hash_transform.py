import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import hashlib


# Custom transformer for hashing
class HashingTransformer(BaseEstimator, TransformerMixin):
    """Hashing transformer for scikit-learn estimators

    Can hash strings, bytes and ints (up to 8 byte ints)
    """
    def __init__(self, hash_func=hashlib.md5, bit_size=128):
        self.hash_func = hash_func
        self.bit_size = bit_size

    @staticmethod
    def hash(item, hash_func, bit_size):
        if isinstance(item, str):
            item = item.encode()
        elif isinstance(item, int):
            item = item.to_bytes(8)
        elif np.issubdtype(item, np.integer):
            item = bytes(item.data)

        if not isinstance(item, bytes):
            raise ValueError('item should be either np.integer, str, bytes or int')

        hash_obj = hash_func(item)
        hashed = hash_obj.digest()
        bit_array = np.unpackbits(np.frombuffer(hashed, dtype=np.uint8))[:bit_size]
        return bit_array

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame)  # TODO: support np.array

        hashed_cols = []

        for col in X.columns:
            hashed_cols.append(np.stack(X[col].apply(self.hash, args=(self.hash_func, self.bit_size))))

        return np.hstack(hashed_cols)
