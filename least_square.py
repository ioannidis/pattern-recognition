import pandas as pd
from sklearn.linear_model import LinearRegression

# TEST DATA
#
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

# GIA ton user1
# Z = ratings_movies.loc[(ratings_movies["user id"] == 1)]
#
# ZZ = Z.values.tolist()
# y = []
#
# for i in range(len(ZZ)):
#     if ZZ[i][2] > 3:
#         y.append(1)
#     else:
#         y.append(-1)
#
# ZZZ = Z.drop(["user id", "movie id", "rating"], axis=1)
# ZZZ = ZZZ.values.tolist()
#
# reg2 = LinearRegression().fit(ZZZ, y)

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


reg = list()
for i in range(len(X)):
    reg.append(LinearRegression().fit(X[i], y[i]))


user_id_input = int(input("Give a user id (1-943):"))
movie_id_input = int(input("Give a movie id (1-1682)"))


selected_movie = movies.loc[(movies["movie id"] == movie_id_input)]
selected_movie = selected_movie.drop("movie id", axis=1)

if (reg[user_id_input - 1].predict(selected_movie)) > 0:
    print("The user has seen this movie")
else:
    print("The user has NOT seen this movie")

