import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(0)

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


"""
Input: Dataset data, number of clusters = k
function which will choose initial cluster representatives or clusters.
Output: Randomly selected centroids within a data in the initial step.
"""
def initialSelection(data, k):
    initial_centroids = np.random.choice(len(data), k, replace = False)   
    centroids = data[initial_centroids, :]
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
    return np.array(clusters) 


"""
Input: Dataset D, centroids = cluster_ids, number of clusters = k.
function, which will compute the cluster representations.
Output: updated cluster assignments. 
"""
def computeClusterRepresentatives(data, centroids, cluster_ids, k):
    for idx in range(k):
        centroids[idx] = np.mean(data[cluster_ids == idx], axis = 0)
    return centroids


"""
Input: Dataset data, number of clusters = k, number of iterations = 100
Kmeans performance method. The method follow this steps:
1. Initialize random cluster centroids
2. Assign each data point to the nearest centroid
3. Update centroids
4. Repeats 2 and 3 until maximum iteration not reached and new centroid points remail unchanged.
Output: data with assigned clusters.
"""
def KMeans(data, k, maxIter=100):
    centroids = initialSelection(data, k)

    for _ in range(maxIter):
        cluster_assignments = assignClusterIds(data, centroids)
        centroids = computeClusterRepresentatives(data, centroids, cluster_assignments, k)
    return np.array(cluster_assignments), np.array(centroids)


"""
function to compute silhouette coefficient to choose right k number. 
Input: Dataset data, cluster representatives =  cluster_assignments
Output: Average Silhouette coefficient
"""
def silhouette_coefficient(data, clusters, k):
    sc_array = []
    
    # Calculate pairwise Euclidean distances
    distances = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            distances[i][j] = np.sqrt(np.sum((data[i] - data[j]) ** 2))
    
    for idx in range(len(data)):
        cluster_id = clusters[idx]
        A = np.mean(distances[idx][np.where(clusters == cluster_id)])
        di = []
        for i in range(k):
            if i == cluster_id:
                continue
            di.append(np.mean(distances[idx][np.where(clusters == i)]))
        B = np.min(di)
        sc_array.append((B - A) / (max(A,B)))
    sc = np.mean(sc_array)
    return sc

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
        cluster_representatives, centroids = KMeans(dataset, k)
        sl_coef = silhouette_coefficient(dataset, cluster_representatives, k)
        s_cs.append(sl_coef)
    x = np.arange(start=1, stop=10, step=1)
    plt.plot(x, s_cs)
    plt.xlabel('Number of clusters k')
    plt.ylabel("Sil coefficient")
    plt.savefig('kmeans.png')

"""
Final plot for task 1. From the plot, silhouette coefficient close to 0, then x is on the border of 2 natural clusters. 
Clustering should be checked to make silhouette coefficient close to 1.
"""
plot_silhouttee()