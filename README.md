Recommendation can be done in multiple ways. One may be choosing items that similar users also like (like is described in the original 
paper published with the dataset). Another way is to use machine learning, where weights are learned, precomputed, from the data, then used
to make predictions.
Which is unlike the forme, non-ML, approach, where computation of the weights and making predictions based off them happens basically at the same time, 
both computed adhoc from the similarity of users or items. In this assignment we will use the latter, ML, approach. 

## Matrix factorization ([retrieval_tfrs.ipynb](retrieval_tfrs.ipynb))
There are multiple ways to do the ML approach. Typically, (looking at the literature/tensorflow recommenders) one would use a matrix factorization approach, 
where the user-item matrix is factorized into two matrices of smaller dimensions, one for users and one for items. The dot product of these two matrices 
then gives the rating, 1 means the user rated the item, and 0 means
the user did not rate the item. (However it is not exactly how it is trained, and the dot product is only computed for the
negative examples in the batch and the matrix is never fully instantiated in its dense form). Then, the closest books to the 
user would be selected as candidates for the recommendation – the best few thousands.

Then a second model is trained, which can be larger, slower and would refine the predictions of the first model. It can make use of
the explicit ratings, too, to refine the order of predictions using the explicit ratings. 

The advantage of this approach is that it is fast to predict; the users and items are represented by a small vector of chosen dimension, and the dot product
is fast to compute with many or all of the items.

I evaluated this approach using top-k accuracy, where true positives are the test items (rated) that were in the top-k of the recommendations, and false positives
are the items that were not in the top-k. (There are no negatives.) 

val_factorized_top_k/top_5_categorical_accuracy: 0.0019  
val_factorized_top_k/top_10_categorical_accuracy: 0.0075  
val_factorized_top_k/top_50_categorical_accuracy: 0.0640  
val_factorized_top_k/top_100_categorical_accuracy: 0.1228  

This is for 6.5K of the books and 64K of test ratings. 

Unfortunately, I haven't recorded this metric before training with initialized random weights, so I don't know how much better it is than random.
But I recorded the validation loss before training, at least in the rating task below, and it did drop significantly 
after training - 21.3 to 12.6 after just one epoch and then slowly increased. So I think the model is better than random.
Also, if there are 6500 books, at random, the probability of a book being in the top-100 is ~1/64, whereas the accuracy is 0.12 ~ 1/8, so it is better than random, around 8x
if we can quantify it like this.

## Rating prediction ([rating_nn.ipynb](rating_nn.ipynb), [rating_linear_and_svr.ipynb](rating_linear_and_svr.ipynb))

Another approach I implemented is to predict the rating directly. A simple regression task. I tried linear regression, SVR and neural network regression.
The results were not impressive. Actually the linear regression has R^2 of 0 on the test set, which means it is not better than predicting a constant – the mean rating
in the test set. The RMSE on test was 3.92, a little better for the SVR, around 3.7. The neural network was the best, with RMSE of 3.58.

The disadvantage of this approach is that it is slow to predict, unlike the user and item vectors, which are precomputed, the prediction
has to be done for each user and item pair. Whereas single dot product for each pair is needed for the matrix factorization approach, in this case 
the entire model has to be run for each pair, which is generally slower. We could subsample the items to be predicted for each user.

SVR took multiple hours to train, since there are around 100K explicit ratings in my filtered dataset. Generally this is 
a large dataset for a SVR which can even have quadratic complexity in the number of samples. Both the linear regression and the neural network
were trained in a few minutes.

Unfortunately, there is no easy way to compare this approach and matrix factorization, since matrix factorization was evaluated using
top-k accuracy, and this approach is evaluated using RMSE. Top-k accuracy would be hard to compute for this approach, since it would require
to compute the rating for #test_users * #books, which is slow.

## OpenAI embedding and adding new users and books

I haven't completed this approach, but there is a gist in  [openai_embeddings.py](openai_embeddings.py). I would use the OpenAI `ada` text embedding model to embed the books and users
into a 1500-dimensional vector space. The idea was to use the cosine similarity to find the closest books to the user. Zero-shot approach.
Or then use this as the input to the matrix factorization approach, leveraging also the relationships between users and books in our dataset.

Their embedding model is trained on a large corpus of text, and the embeddings are trained to be similar for words that are similar in meaning. 
The books are embedded as a string of their metadata - the tile, author, year, etc. The user is embedded as a concatenation
of the books metadata the user rated (possibly subsampled), plus his location and age.

The advantage of this approach is that it is fast to predict, similar to the matrix factorization approach. However, we cannot control the 
dimensionality of the embedding space, which is rather large. In production, we could use vector databases, which use some 
heuristics to compute nearest neighbors. If we used the embedding as the input to the matrix factorization approach,
we could to reduce the dimensionality of the embedding, therefore possibly making the prediction faster.

The main advantage of the OpenAI approach is that it is zero-shot, we don't need to train the model on our dataset, we can just use the pre-trained model. And as new
user and book metadata is added to the dataset, no training needs to be done, and it can be used immediately, even if we add the matrix-factorization approach to the OpenAI embeddings.
Unfortunately, in the other approaches mentioned in the former sections, the model would have to be retrained, since the users and books are represented by an integer index, respectively the
one-hot encoding of the index; the model would have to be retrained to take into account the new users and books. Or other workarounds would be needed, such
as passing as new user's recommendations the recommendations of the most similar users in the dataset.


## More complicated solutions

Also in the matrix factorization approach, the neural network model can be trained multi-task, for retrieval (implicit ratings mainly) and ranking ("sorting") tasks
at once. This is not done in this assignment, but it is possible to do it.

There are also reinforcement learning approaches, where the time is taken into account, and the user is modeled as a Markov Decision Process, making sequential
actions. The predictions could therefore be influenced by the previous actions, and the model could learn to recommend items that are not only _relevant_, but also
diverse, or reflect the user's mood or _interests evolving over time_.
