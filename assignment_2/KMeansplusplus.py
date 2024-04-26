import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(0)

"""
Loading and normalizing data. Dataset is vector representations of words on 
initial column[0].
"""
def load_dataset(file='dataset'):
    word_vec = {}
    try:
        with open(file) as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                word_vec[word] = vector
        dataset = np.array(list(word_vec.values()))

        if len(dataset) < 2:
            raise ValueError("Dataset contains less than 2 data")
    except FileNotFoundError:
        print(f"File '{file}' not found")
        return None
    except ValueError:
        print(f"File '{file}' is corrupted or error file")
        return None
    return dataset

"""
function which will choose initial cluster representatives or clusters. 
Function will choose centroids by k-means++. Input for function is dataset and number of clusters.
The output is centroids for clustering. this function follows this steps:
1. Initialisation of first cluster randomly by selecting one point from dataset;
2. For each point (except centroid points) calculate distance between the point and the closest cluster center;
3. Choose next centroid point from the remaining points with probability proportional to the distance squared to the nearest centroid;
"""
def initialSelection(data, k):
    centroids = [data[np.random.choice(len(data))]]

    while len(centroids) < k:
        distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in data])
        prob = distances**2 / np.sum(distances**2)
        new_centroid = data[np.random.choice(len(data), p=prob)]
        centroids.append(new_centroid)
    return centroids


"""
Input: 2 points
function to computing the distance between two points, distance calculated by Euclidean distance.
Output: Euclidean distance
"""
def ComputeDistance(vec_1, vec_2):
    temp = vec_1 - vec_2
    distance = np.sqrt(np.dot(temp.T, temp))
    return distance


"""
Input: Dataset data, centroids
function that will assign cluster ids to each data point.
Output: Updated Centroids indexes to cluster data
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
Input: Dataset D, centroids = cluster_ids, number of clusters = k.
function, which will compute the cluster representations.
Output: updated cluster assignments. 
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
def KMeanspp(data, k, maxIter=100):
    centroids = initialSelection(data, k)

    for _ in range(maxIter):
        cluster_assignments = assignClusterIds(data, centroids)
        new_centroids = computeClusterRepresentatives(data, cluster_assignments, k)
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return cluster_assignments


"""
function to compute silhouette coefficient to choose right k number. 
Input: Dataset data, cluster representatives =  cluster_assignments
Output: Average Silhouette coefficient
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
Input: without input as parameters. But calls methods, load_dataset to load dataset
KMeans to get the clustersIds, silhouette_coefficient 
to compute Silhouette coefficient with number of clusters 1 to 9. 
Output: Plot of vertical silhouette_coefficient and horizontal number of clusters.
"""
def plot_silhouttee():
    dataset = load_dataset()
    s_cs = []
    for k in range(1, 10):
        cluster_representatives = KMeanspp(dataset, k)
        sl_coef = silhouette_coefficient(dataset, cluster_representatives)
        s_cs.append(sl_coef)
    x = np.arange(start=1, stop=10, step=1)
    plt.plot(x, s_cs)
    plt.xlabel('Number of clusters k')
    plt.ylabel("Sil coefficient")
    plt.savefig('kmeansplusplus.png')

"""
Final plot for task 3. From the plot, silhouette coefficient close to 0, then x is on the border of 2 natural clusters. 
Clustering should be checked to make silhouette coefficient close to 1.
"""
plot_silhouttee()