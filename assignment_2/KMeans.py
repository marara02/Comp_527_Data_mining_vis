import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(0)

"""
Loading and normalizing data. Dataset is vector representations of words on 
initial column[0]. Normalization calculates vector norms by computing
Euclidean norm of each vector
"""
def load_dataset():
    word_vec = {}
    with open('dataset') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            word_vec[word] = vector
    dataset = np.array(list(word_vec.values()))

    # Normalization of word vectors
    dataset_normalized = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
    return dataset_normalized

"""
function which will choose initial cluster representatives or clusters. ?? Should ask about functionality
"""
def initialSelection(data, k):
    centroids = []
    for i in range(k):
        centroid = data[random.randint(0, 149)]
        centroids.append(centroid)
    return centroids

# 
"""
function to computing the distance between two points, distance calculated by Euclidean distance.
"""
def ComputeDistance(vec_1, vec_2):
    temp = vec_1 - vec_2
    distance = np.sqrt(np.dot(temp.T, temp))
    return distance

# function, where x is the data and k is the value of maxIter.
def clustername(x,k):
    return x

"""
function that will assign cluster ids to each data point.
"""
def assignClusterIds(data, centroids):
    cluster_assignments = []

    for d in data:
        dist_point_clust = []
        for centroid in centroids:
            distance = ComputeDistance(d, centroid)
            dist_point_clust.append(distance)
        
        assignment = np.argmin(dist_point_clust)
        cluster_assignments.append(assignment)

    return cluster_assignments   

"""
function, which will compute the cluster representations.
"""
def computeClusterRepresentatives(data, cluster_ids, k):
    cluster_representatives = []
    for cluster_id in range(k):
        cluster_points = data[np.array(cluster_ids) == cluster_id]
        if len(cluster_points) > 0:
            cluster_representative = np.mean(cluster_points, axis=0)
            cluster_representatives.append(cluster_representative)
    return cluster_representatives

"""
Kmeans performance method. The method follow this steps:
1. Initialize random cluster centroids
2. Assign each data point to the nearest centroid
3. Update centroids
4. Repeats 2 and 3 until maximum iteration not reached and new centroid points remail unchanged.
"""
def KMeans(data, k, maxIter=100):
    centroids = initialSelection(data, k)

    for _ in range(maxIter):
        cluster_assignments = assignClusterIds(data, centroids)
        new_centroids = computeClusterRepresentatives(data, cluster_assignments, k)
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, cluster_assignments

"""
function to compute silhouette coefficient to choose right k number. 
"""
def silhouette_coefficient(data, cluster_assignments):
    n = len(data)
    silhouette_vals = []

    # Compute distance matrix
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(data[i] - data[j])
    
    # Condition if the number of cluster is 1, return 0 as silhouette coefficient
    if len(set(cluster_assignments)) == 1:
        return 0

    for i in range(n):
        cluster_idx = cluster_assignments[i]
        cluster_points = [idx for idx, c in enumerate(cluster_assignments) if c == cluster_idx]

        # average distance from i to all other points in the same cluster
        A = np.mean([distances[i, j] for j in cluster_points if j != i])
        # smallest average distance from i to all points in different clusters
        B = np.min([np.mean([distances[i, j] for j in range(n) if cluster_assignments[j] != cluster_idx]) for cluster_idx in set(cluster_assignments) if cluster_idx != cluster_assignments[i]])
       
        s_cf = (B - A) / max(A, B)
        silhouette_vals.append(s_cf)

    silhouette_avg = np.mean(silhouette_vals)
    return silhouette_avg

"""
function to plot number of clusters vs. silhouttee coefficient values.
"""
def plot_silhouttee():
    dataset = load_dataset()
    s_cs = []
    centroids, cluster_representatives = KMeans(dataset, 1)
    for k in range(1, 11):
        centroids, cluster_representatives = KMeans(dataset, k)
        sl_coef = silhouette_coefficient(dataset, cluster_representatives)
        s_cs.append(sl_coef)
    x = np.arange(10)
    plt.plot(x, s_cs)
    plt.xlabel('Number of clusters')
    plt.ylabel("Sil coefficient")
    plt.show()

"""
Final plot for task 1. 
From the plot, silhouette coefficient close to 0, then x is on the border of 2 natural clusters. 
Clustering should be checked to make silhouette coefficient close to 1.
"""
plot_silhouttee()