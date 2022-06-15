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
    print("mlalgo:")
    print("\tscore: ", round(ms.score(np.array([[2,1.5],[4.5,4]])),2))
    print("\tinertia: ", round(ms.inertia,1))
    print("----------------------------")
    # Finally We plot the data points and centroids
    fig, axs = plt.subplots(1,2)

    axs[0].scatter(X[:, 0], X[:, 1],marker ='o',c=ms.labels)
    axs[0].scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker ='x', color ='red',
    		s = 300, linewidth = 5, zorder = 1)
    axs[0].set_title("mlalgo")
    
    # sklearn
    ms = KM(5,init="random").fit(X)
    cluster_centers = ms.cluster_centers_
    
    print("sklearn:")
    print("\tscore: ", round(ms.score(np.array([[2,1.5],[4.5,4]])),2))
    print("\tinertia: ", round(ms.inertia_,1))

    axs[1].scatter(X[:, 0], X[:, 1],marker ='o',c=ms.labels_)
    axs[1].scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker ='x', color ='red',
    		s = 300, linewidth = 5, zorder = 1)
    axs[1].set_title("sklearn")

    plt.show()

if __name__ == "__main__":
    main()