from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from movielens_data import MovieLensData

data = MovieLensData()

users = data.ratings
movies = data.movies

user_count = 943
movies_count = 1682

X = np.zeros((user_count, movies_count))

for user in range(user_count):
    X[user] = np.arange(1, movies_count + 1)

for user in range(user_count):
    movie_ids = users[(users['user id'] == user)]['movie id'].values

y = np.zeros((user_count, movies_count))

for user in range(1, user_count + 1):
    movies_for_user = users[(users['user id'] == user)]['movie id'].values
    for movie in movies_for_user:
        y[user - 1][movie - 1] = 1

# y = y.reshape(user_count * movies_count)

X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train = normalize(X_train)
X_test = normalize(X_test)

neural_network = MLPClassifier(
    hidden_layer_sizes=(20,),
    activation='logistic',
    max_iter=100)

neural_network.fit(X_train, y_train)
