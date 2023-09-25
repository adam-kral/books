from typing import Union

import pandas as pd
from openai.embeddings_utils import get_embedding, distances_from_embeddings, indices_of_nearest_neighbors_from_distances


book_embedding_template = '''
Book-Title: {title}
Book-Author: {author}
Year-Of-Publication: {year}
Publisher: {publisher}
'''


rated_book_embedding_template = '''
{book_embedding}
Book-Rating: {rating}
'''

user_embedding_template = '''
User:
Location: {location}
Age: {age}
Total-Ratings: {total_ratings}
Some-Rated-Books: {some_rated_books}
'''


def embed_book(book: Union[pd.Series, dict]):
    """Return an embedding for a book."""

    book_str = book_embedding_template.format(
        title=book["Book-Title"],
        author=book["Book-Author"],
        year=book["Year-Of-Publication"],
        publisher=book["Publisher"],
    )

    return get_embedding(book_str, model="text-embedding-ada-002")


MAX_BOOKS_PER_USER = 15


def embed_books(books_df):
   """Return embeddings for book titles."""

   return [embed_book(book) for book in books_df.iterrows()]


def embed_users(users_df, ratings_df):
    """Return embeddings for users."""

    ratings_df = ratings_df.set_index("User-ID")

    embeddings = []
    for user in users_df.iterrows():
        user_ratings = ratings_df.loc[user["User-ID"]]
        selected_books = user_ratings.sample(min(MAX_BOOKS_PER_USER, len(user_ratings)), random_state=user["User-ID"])

        selected_books_str = "\n".join(
            [
                rated_book_embedding_template.format(
                    book_embedding=embed_books(book),
                    rating=book["Book-Rating"],
                )
                for book in selected_books
            ]
        )

        user_str = user_embedding_template.format(
            location=user["Location"],
            age=user["Age"],
            total_ratings=user["user_ratings_count"],
            some_rated_books=selected_books_str,
      )

        embeddings.append(get_embedding(user_str, model="text-embedding-ada-002"))

    return embeddings


# from openai tutorial, ideas for library functions I could use (or just do it in numpy)
def recommendations_from_strings(
   strings: list[str],
   index_of_source_string: int,
   model="text-embedding-ada-002",
) -> list[int]:
   """Return nearest neighbors of a given string."""

   # get embeddings for all strings (better to do in batches, but this is just a demo)
   embeddings = [get_embedding(string, model=model) for string in strings]

   # get the embedding of the source string
   query_embedding = embeddings[index_of_source_string]

   # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
   distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")

   # get indices of nearest neighbors (function from embeddings_utils.py)
   indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
   return indices_of_nearest_neighbors