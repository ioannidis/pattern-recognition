# Pattern Recognition 2018-2019  #
#================================#
# p16036 - Ioannidis Panagiotis  #
# p16112 - Paravantis Athanasios #
#================================#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# pd.set_option('display.max_columns', 500)

#  TESTING PANDAS ========================================================================
data = [[2.5, 3.5], [2.5, 2.5], [-1, -1.5], [0.8, 0.8]]
theta_list = np.arange(24, 72, 0.1)
print(theta_list)

u_cols = ["user id" , "movie id" , "rating", "timestamp"]
ratings = pd.read_csv('./u.data', sep='\t', names=u_cols, encoding='latin-1')

ratings = ratings.sort_values("user id")
ratings = ratings.loc[(ratings["rating"] >= 4)]
# print(ratings)
# print(len(ratings))

# Extract data using pandas
u_cols = ["movie id" , "movie title", "release date" , "video release date" ,
              "IMDb URL" , "unknown" , "Action" , "Adventure" , "Animation" ,
              "Children's" , "Comedy" , "Crime" , "Documentary"
              , "Drama" , "Fantasy" ,
              "Film-Noir" , "Horror" , "Musical" , "Mystery" , "Romance" , "Sci-Fi" ,
              "Thriller" , "War" , "Western"]

movies = pd.read_csv('./u.item', sep='|', names=u_cols, encoding='latin-1')
movies = movies.iloc[:, [0] + list(range(6, 24))]
# print(movies)

newArr = pd.merge(ratings,movies, on='movie id')
newArr = newArr.sort_values("user id")
newArrOnlyRatings = newArr.iloc[:, [0] + list(range(4, 22))]
print(newArrOnlyRatings)

# data = np.genfromtxt('./data1N.csv', delimiter=',')

# Data  values
# result = newArrOnlyRatings.iloc[:, list(range(1, 19))]
# data = result.values.tolist()
# print(data[0])
# end Data  values


# Data sum values
result = newArrOnlyRatings.groupby(["user id"]).sum()
result = result.iloc[:, list(range(0, 18))]
print(result)
data = result.values.tolist()
# print(data)
# end Data sum values

# data = newArrOnlyRatings.values.tolist()
# print(data)

#print(newArrOnlyRatings.values.tolist())



# data = movies.iloc[:,5:24].values.tolist()

# END TESTING PANDAS ========================================================================

# BSAS =========================================================================================

def bsas(data, theta):
    m                   = 1
    clusters            = []
    cluster_centroids   = []

    clusters.append(data[0])
    cluster_centroids.append(data[0])

    # print(clusters)
    # print(cluster_centroids)

    for vector in data[1:]:
        euclidean_distances = []
        # print("next vector: ", vector)

        for centroid in cluster_centroids:
            euclidean_distances.append(np.linalg.norm(np.subtract(vector, centroid)))

        min_euclidean_distance_pos = np.argmin(euclidean_distances)
        # print("euclidean_distances: ", euclidean_distances)
        # print("min_euclidean_distance_pos: ", min_euclidean_distance_pos)
        if euclidean_distances[min_euclidean_distance_pos] > theta:
            m += 1
            clusters.append(vector)
            cluster_centroids.append(vector)
        else:
            clusters[min_euclidean_distance_pos]            = [clusters[min_euclidean_distance_pos], vector]
            cluster_centroids[min_euclidean_distance_pos]   = calculate_centroid(len(clusters[min_euclidean_distance_pos]), cluster_centroids[min_euclidean_distance_pos], vector).tolist()

        # print("=======================================================================================")

    # print("clusters: " , clusters)
    print("cluster centroids: ", cluster_centroids)
    return [clusters, cluster_centroids]

def calculate_centroid(cluster_size, centroid, vector):
    # print("calculate_centroid parameters: ", cluster_size, centroid, vector)
    # print("calculate_centroid: ", np.true_divide(np.add(np.multiply(cluster_size - 1, centroid), vector), cluster_size))
    return np.add(np.multiply(cluster_size - 1, centroid), vector) / cluster_size


def clusters_number_estimation(theta_list, data):
    cluster_estimations = []

    np.random.shuffle(data)

    for theta in theta_list:
        # print("shuffled data: ", data)
        bsas_result = bsas(data, theta)
        cluster_estimations.append([theta, len(bsas_result[0])])


    # print(cluster_estimations)

    cluster_number_for_each_theta = [number[1] for number in cluster_estimations]
    fig, ax = plt.subplots()
    ax.plot(theta_list, cluster_number_for_each_theta )
    ax.set(xlabel='Theta', ylabel='Number of clusters')
    ax.grid()
    plt.show()



clusters_number_estimation(theta_list, data)

# TODO: TESTING
# bsas([[2.5, 3.5], [2.5, 2.5], [-1, -1.5], [0.8, 0.8]], 2)
# clusters_number_estimation(theta_list, [[2.5, 3.5], [2.5, 2.5], [-1, -1.5], [0.8, 0.8]])
# clusters_number_estimation(theta_list, [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# bsas([[-1, -1.3], [0.7, 0.8], [2, 3], [2.5, 2.5]], 0.6)
