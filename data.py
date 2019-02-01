# Pattern Recognition 2018-2019  #
#================================#
# p16036 - Ioannidis Panagiotis  #
# p16112 - Paravantis Athanasios #
#================================#

import pandas as pd

class Data:

    def __init__(self):
        self.load_user_ratings_data()
        self.load_movies_data()
        self.merge_ratings_movies()
        self.movie_categories_liked_per_user()

    def load_user_ratings_data(self):
        # Load user rating data
        r_cols = ["user id" , "movie id" , "rating", "timestamp"]
        ratings = pd.read_csv('./u.data', sep = '\t', names = r_cols, encoding = 'latin-1')
        # Sort ratings by user id
        ratings = ratings.sort_values("user id")
        # Keep rating greater equal than 4
        self.ratings = ratings.loc[(ratings["rating"] >= 4)]

    def load_movies_data(self):
        # Load movies data
        m_cols = ["movie id", "movie title", "release date", "video release date", "IMDb URL", "unknown", "Action",
                  "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                  "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        movies = pd.read_csv('./u.item', sep = '|', names = m_cols, encoding = 'latin-1')
        # Keep movie id and movie categories (except 'unknown')
        self.movies = movies.iloc[:, [0] + list(range(6, 24))]

    def merge_ratings_movies(self):
        # Merge ratings dataset with movies dataset on movie id
        ratings_movies = pd.merge(self.ratings, self.movies, on = 'movie id')
        self.ratings_movies = ratings_movies.sort_values("user id")

    def movie_categories_liked_per_user(self):
        # Keep user id and movies categories
        return self.ratings_movies.iloc[:, [0] + list(range(4, 22))]

    def sum_movie_categories_liked_per_user(self):
        # Sum movie each movie category for each user
        sum_movie_categories_liked_per_user = self.movie_categories_liked_per_user().groupby(["user id"]).sum()
        # Keep only the movie categories
        return sum_movie_categories_liked_per_user.iloc[:, list(range(0, 18))]


    # Data normalization
    # Calculate min - max values for each user
    def calculate_min_max(self):
        min_max = pd.DataFrame([])
        min_max['min_value'] = self.sum_movie_categories_liked_per_user().min(axis = 1)
        min_max['max_value'] = self.sum_movie_categories_liked_per_user().max(axis = 1)

        return min_max

    def get_normalized_data(self):
        results = self.sum_movie_categories_liked_per_user().values.tolist()
        norm_list = self.calculate_min_max().values.tolist()

        for index, tuple in enumerate(results):
            for i in range(len(tuple)):
                tuple[i] = (tuple[i] - norm_list[index][0]) / (norm_list[index][1] - norm_list[index][0])

        return results
