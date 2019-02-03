from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from movielens_data import MovieLensData

data = MovieLensData()

users = data.ratings
movies = data.movies

user_count = 943
movies_count = 1682

print('Creating input data (X)...')
X = [[user_id + 1, movie_id + 1] for user_id in range(user_count) for movie_id in range(movies_count)]

print('Creating labels (y)...')
y = []

for user_id in range(1, user_count + 1):
    seen_movies = [0 for i in range(movies_count)]
    query = users[users['user id'] == user_id]['movie id'].tolist()
    for movie_id in query:
        seen_movies[movie_id - 1] = 1
    y = y + seen_movies

print('Creating training and test splits...')
X_train, X_test, y_train, y_test = train_test_split(X, y)

print('Normalizing training data...')
X_train = normalize(X_train)
X_test = normalize(X_test)

print('Initializing neural network...')
neural_network = MLPClassifier(
    hidden_layer_sizes=(20,),
    activation='logistic',
    max_iter=100)

print('Training neural network...')
neural_network.fit(X_train, y_train)

print('5-fold cross validation...')
print(cross_val_score(neural_network, X_test, y_test, cv=5))
