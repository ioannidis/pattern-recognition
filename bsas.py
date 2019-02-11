# Pattern Recognition 2018-2019  #
#================================#
# p16036 - Ioannidis Panagiotis  #
# p16112 - Paravantis Athanasios #
#================================#

import numpy as np
import matplotlib.pyplot as plt
from movielens_data import MovieLensData

dataObj = MovieLensData()
data = dataObj.get_normalized_data()

# Basic sequential algorithmic scheme
def bsas(data, theta):
    m                   = 1
    clusters            = []
    cluster_centroids   = []

    clusters.append(data[0])
    cluster_centroids.append(data[0])

    for vector in data[1:]:
        euclidean_distances = []

        for centroid in cluster_centroids:
            euclidean_distances.append(np.linalg.norm(np.subtract(vector, centroid)) ** 2)

        min_euclidean_distance_pos = np.argmin(euclidean_distances)

        if euclidean_distances[min_euclidean_distance_pos] > theta:
            m += 1
            clusters.append(vector)
            cluster_centroids.append(vector)
        else:
            clusters[min_euclidean_distance_pos]            = [clusters[min_euclidean_distance_pos], vector]
            cluster_centroids[min_euclidean_distance_pos]   = calculate_centroid(len(clusters[min_euclidean_distance_pos]), cluster_centroids[min_euclidean_distance_pos], vector).tolist()

    return [clusters, cluster_centroids]

def calculate_centroid(cluster_size, centroid, vector):
    return np.add(np.multiply(cluster_size - 1, centroid), vector) / cluster_size


def clusters_number_estimation(theta_min, theta_max, theta_step, data):
    theta_list = np.arange(theta_min, theta_max, theta_step)
    cluster_estimations = []

    np.random.shuffle(data)

    for theta in theta_list:
        bsas_result = bsas(data, theta)
        cluster_estimations.append([theta, len(bsas_result[0])])

    with open('result.txt', 'w') as f:
        for i in range(len(cluster_estimations)):
            f.write("%.3f %.1f \n" % (theta_list[i], cluster_estimations[i][1]))


    cluster_number_for_each_theta = [number[1] for number in cluster_estimations]
    fig, ax = plt.subplots()
    ax.plot(theta_list, cluster_number_for_each_theta )
    ax.set(xlabel='Theta', ylabel='Number of clusters')
    ax.grid()
    plt.show()



clusters_number_estimation(0.1 , 2.1, 0.005, data)

