import pandas as pd
from sklearn.neural_network import MLPClassifier
from movielens_data import MovieLensData

data = MovieLensData()
fold_number = 5

data_train = data.load_fold_data(fold_number, "base")
movies = data.movies

ratings_movies_train = pd.merge(data_train, movies, on='movie id')
ratings_movies_train = ratings_movies_train.sort_values(["user id", "rating"])


X = list()
for i in range(1, 944):
    X.append(ratings_movies_train.loc[(ratings_movies_train["user id"] == i)])

y = list()
for i in range(len(X)):
    yy = list()
    for index, row in X[i].iterrows():
        if row["rating"] > 3:
            yy.append(1)
        else:
            yy.append(-1)
    y.append(yy)
    X[i] = X[i].drop(["user id", "movie id", "rating"], axis=1)


data_test = data.load_fold_data(fold_number, "test")

ratings_movies_test = pd.merge(data_test, movies, on='movie id')
ratings_movies_test = ratings_movies_test.sort_values(["user id", "rating"])


X_test = list()
for i in range(1, 944):
    X_test.append(ratings_movies_test.loc[(ratings_movies_test["user id"] == i)])

for i in range(len(X_test)):
    X_test[i] = X_test[i].drop(["user id", "movie id", "rating"], axis=1)


clf = list()
for i in range(len(X)):
    clf.append(MLPClassifier(hidden_layer_sizes=(18, 9, 18), activation='logistic', solver='lbfgs').fit(X[i], y[i]))


while True:

    while True:
        try:
            user_id_input = int(input("Give a user id (1-943):"))
        except ValueError:
            print("Please, type a valid number!")
            continue

        if user_id_input < 1 or user_id_input > 943:
            print("Please, type a number between 1 and 943.")
            continue
        else:
            break

    while True:
        try:
            movie_id_input = int(input("Give a movie id (1-1682)"))
        except ValueError:
            print("Please, type a valid number!")
            continue

        if movie_id_input < 1 or movie_id_input > 1682:
            print("Please, type a number between 1 and 1682.")
            continue
        else:
            break

    selected_movie = movies.loc[(movies["movie id"] == movie_id_input)]
    selected_movie = selected_movie.drop("movie id", axis=1)

    if (clf[user_id_input - 1].predict(selected_movie)) > 0:
        print("The user has seen this movie")
    else:
        print("The user has NOT seen this movie")

