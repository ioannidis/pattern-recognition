from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from movielens_data import MovieLensData

# Neural Network and Least Squares classification

# Load all user ratings and movie details
data = MovieLensData()
users = data.ratings
movies = data.movies

# The number of users and movies on the dataset
user_count = 943
movies_count = 1682

print('Creating input data (X)...')

# Create an 2d array for each: [..., [user id, movie id], ...]
X = [[user_id + 1, movie_id + 1] for user_id in range(user_count) for movie_id in range(movies_count)]

print('Creating labels (y)...')

# Create a 1d array of labels for each entry on array X
y = []

# Create the y array made out of 0s and 1s
# 0 if a user hasn't seen a movie
# 1 if a user has seen (rated) a movie
for user_id in range(1, user_count + 1):
    seen_movies = [0 for i in range(movies_count)]
    query = users[users['user id'] == user_id]['movie id'].tolist()
    for movie_id in query:
        seen_movies[movie_id - 1] = 1
    y = y + seen_movies

print('Creating training and test splits...')
X_train, X_test, y_train, y_test = train_test_split(X, y)

# print('Normalizing training data...')
# X_norm_train = normalize(X_train)
# X_norm_test = normalize(X_test)

print('Initializing neural network...')

neural_network = MLPClassifier(
    hidden_layer_sizes=(20,),
    activation='logistic',
    max_iter=100)

print('Training neural network...')
neural_network.fit(X_train, y_train)

print('5-fold cross validation...')
print(cross_val_score(neural_network, X_test, y_test, cv=5))

print('Initializing least squares...')
least_squares = LinearRegression()

print('Training least squares...')
least_squares.fit(X_train, y_train)
