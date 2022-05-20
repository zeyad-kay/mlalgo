from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets._samples_generator import make_blobs
from mlalgo.cluster import KMeans
from sklearn.cluster import KMeans as KM

def main():
    # We will be using the make_blobs method
    # in order to generate our own data.
    clusters = [[2, 2],[4,4],[8,14],[8,8],[8,2]]

    X, _ = make_blobs(n_samples = 1000, centers = clusters,
    								cluster_std = 0.60)

    # mlalgo
    ms = KMeans(5).fit(X)
    cluster_centers = ms.centers
    print("Score: ", ms.score(np.array([[2,1.5],[4.5,4]])))
    print("Inertia: ", ms.inertia)
    
    # Finally We plot the data points and centroids
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.scatter(X[:, 0], X[:, 1],marker ='o',c=ms.labels)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker ='x', color ='red',
    		s = 300, linewidth = 5, zorder = 1)

    plt.show()
    
    # sklearn
    ms = KM(5,init="random").fit(X)
    cluster_centers = ms.cluster_centers_
    print("Score: ", ms.score(np.array([[2,1.5],[4.5,4]])))
    print("Inertia: ", ms.inertia_)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.scatter(X[:, 0], X[:, 1],marker ='o',c=ms.labels_)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker ='x', color ='red',
    		s = 300, linewidth = 5, zorder = 1)

    plt.show()

if __name__ == "__main__":
    main()