import numpy as np
import matplotlib.pyplot as plt
import random


"""
Loading and normalizing data. Dataset is vector representations of words on 
initial column[0]. Normalization calculates vector norms by computing
Euclidean norm of each vector
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
        # Normalization of word vectors
        dataset_normalized = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
    except FileNotFoundError:
        print(f"File '{file}' not found")
        return None
    except ValueError:
        print(f"File '{file}' is corrupted or error file")
        return None
    return dataset_normalized


# Coverting array to numpy array
def convert_data_npArray(data):
    data = np.array(data)
    if len(data.shape) == 1:
        data = np.expand_dims(data, -1)
    return data


"""
Input: Dataset data, number of clusters = k
function which will choose initial cluster representatives or clusters.
Output: Randomly selected centroids within a data in the initial step.
"""
def initialSelection(data, k):
    centroids = []
    while len(centroids) < k:
        centroid = data[random.randint(0, 149)]
        centroids.append(centroid)
    return centroids


"""
Input: 2 points
function to computing the distance between two points, distance calculated by Euclidean distance.
Output: Euclidean distance
"""
def ComputeDistance(vec_1, vec_2):
    return np.linalg.norm(vec_1 - vec_2, axis=0)


"""
Input: Dataset data, centroids
function that will assign cluster ids to each data point.
Output: Updated Centroids indexes to cluster data
"""
def assignClusterIds(data, centroids):
    clusters = []
    for point in data:
        distances = [ComputeDistance(point, centroid) for centroid in centroids]
        cluster_id = distances.index(min(distances))
        clusters.append(cluster_id)
    return clusters  


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
def KMeans(data, k, maxIter=100):
    centroids = initialSelection(data, k)
    for _ in range(maxIter):
        cluster_assignments = assignClusterIds(data, centroids)
        new_centroids = computeClusterRepresentatives(data, cluster_assignments, k)
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return cluster_assignments

"""
Function to compute the sum of squared distances within a cluster
Input: Data Points
ss_distance = sum(dist(points, centroid)^2)
Output is the sum of squared distances within a cluster
"""
def computeSumfSquare(points):
    centroids = np.mean(points, 0)
    ss_distance = np.linalg.norm(points - centroids, ord=2, axis=1)
    return np.sum(ss_distance)

"""
Function to clustering by Bisecting algorithm
Input: Dataset data, number of clusters = k
Output: return the leaf clusters
1. Initializing tree with whole Dataset data
2. Repeat steps
   - Select leaf node L that has the max the sum of squared distances within a cluster
   - Split L node child leaves into clusters using KMeans
   - Add leaves as children of L in tree
3. Stop when then number of leaf clusters is k
"""
def BisectingKMeans(data, k):
    data = convert_data_npArray(data)
    clusters = [data]
    while len(clusters) < k:
        max_sse_index = np.argmax([computeSumfSquare(c) for c in clusters])
        cluster = clusters.pop(max_sse_index)
        k_clusters = KMeans(cluster, 9)
        clusters.extend(k_clusters)
    return clusters

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
BisectingKMeans to get the leaf clusters, silhouette_coefficient 
to compute Silhouette coefficient with number of clusters 1 to 9. 
Output: Plot of vertical silhouette_coefficient and horizontal number of clusters.
"""
def plot_silhouttee():
    dataset = load_dataset()
    s_cs = []
    for k in range(2, 10):
        cluster_representatives = BisectingKMeans(dataset, k)
        sl_coef = silhouette_coefficient(dataset, cluster_representatives)
        s_cs.append(sl_coef)
    x = np.arange(start=2, stop=10, step=1)
    plt.plot(x, s_cs)
    plt.xlabel('Number of clusters k')
    plt.ylabel("Sil coefficient")
    plt.savefig('bisectingkmeans.png')

plot_silhouttee()
