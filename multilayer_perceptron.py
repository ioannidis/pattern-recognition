import pandas as pd
from sklearn.linear_model import LinearRegression

# TEST DATA
#
from sklearn.neural_network import MLPClassifier

r_cols_1 = ["user id" , "movie id" , "rating", "timestamp"]
train1 = pd.read_csv('./u1.base', sep = '\t', names = r_cols_1, encoding = 'latin-1')
# Sort ratings by user id
train1 = train1.drop("timestamp", axis=1)
train1 = train1.sort_values("user id")

m_cols = ["movie id", "movie title", "release date", "video release date", "IMDb URL", "unknown", "Action",
                  "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                  "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
movies = pd.read_csv('./u.item', sep = '|', names = m_cols, encoding = 'latin-1')
# Keep movie id and movie categories (except 'unknown')
movies = movies.iloc[:, [0] + list(range(6, 24))]

ratings_movies = pd.merge(train1, movies, on='movie id')
ratings_movies = ratings_movies.sort_values(["user id", "rating"])


X = list()
for i in range(1, 944):
    X.append(ratings_movies.loc[(ratings_movies["user id"] == i)])

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



clf = []

for i in range(len(X)):
    clf.append(MLPClassifier(hidden_layer_sizes=(100, 50, 100), activation='logistic', solver='lbfgs').fit(X[i], y[i]))

print("end")

[[0 0 0 ... 0 0 0]
 [1 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]