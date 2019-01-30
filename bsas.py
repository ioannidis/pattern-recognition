# Pattern Recognition 2018-2019  #
#================================#
# p16036 - Ioannidis Panagiotis  #
# p16112 - Paravantis Athanasios #
#================================#

import numpy as np
import matplotlib.pyplot as plt

data = ""
theta_list = np.arange(1, 21, 1)

# Extract data using pandas


def bsas(data, theta):
    m                   = 1
    clusters            = []
    cluster_centroids   = []

    clusters.append(data[0])
    cluster_centroids.append(data[0])

    print(clusters)
    print(cluster_centroids)

    for vector in data[1:]:
        euclidean_distances = []
        print("next vector: ", vector)

        for centroid in cluster_centroids:
            euclidean_distances.append(np.linalg.norm(np.subtract(vector, centroid)) ** 2)

        min_euclidean_distance_pos = np.argmin(euclidean_distances)
        print("euclidean_distances: ", euclidean_distances)
        print("min_euclidean_distance_pos: ", min_euclidean_distance_pos)
        if euclidean_distances[min_euclidean_distance_pos] > theta:
            m += 1
            clusters.append(vector)
            cluster_centroids.append(vector)
        else:
            clusters[min_euclidean_distance_pos]            = [clusters[min_euclidean_distance_pos], vector]
            cluster_centroids[min_euclidean_distance_pos]   = calculate_centroid(len(clusters[min_euclidean_distance_pos]), cluster_centroids[min_euclidean_distance_pos], vector).tolist()

        print("=======================================================================================")

    print("clusters: " ,clusters)
    print("cluster centroids: ", cluster_centroids)
    return [clusters, cluster_centroids]

def calculate_centroid(cluster_size, centroid, vector):
    print("calculate_centroid parameters: ", cluster_size, centroid, vector)
    print("calculate_centroid: ", np.true_divide(np.add(np.multiply(cluster_size - 1, centroid), vector), cluster_size))
    return np.add(np.multiply(cluster_size - 1, centroid), vector) / cluster_size


def clusters_number_estimation(theta_list, data):
    cluster_estimations = []
    np.random.shuffle(data)
    print("shuffled data: ", data)
    for theta in theta_list:
        bsas_result = bsas([[2.5, 3.5], [2.5, 2.5], [-1, -1.5], [0.8, 0.8]], theta)
        cluster_estimations.append([theta, len(bsas_result[0])])

    print(cluster_estimations)

    cluster_number_for_each_theta = [number[1] for number in cluster_estimations]
    fig, ax = plt.subplots()
    ax.plot(theta_list, cluster_number_for_each_theta )
    ax.set(xlabel='Theta', ylabel='Number of clusters')
    ax.grid()
    plt.show()



bsas([[2.5, 3.5], [2.5, 2.5], [-1, -1.5], [0.8, 0.8]], 2)
clusters_number_estimation(theta_list, [[2.5, 3.5], [2.5, 2.5], [-1, -1.5], [0.8, 0.8]])
# bsas([[-1, -1.3], [0.7, 0.8], [2, 3], [2.5, 2.5]], 0.6)
