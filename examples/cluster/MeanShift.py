from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from mlalgo.cluster import MeanShift

def main():
    # We will be using the make_blobs method
    # in order to generate our own data.
    clusters = [[2, 2],[8,14],[8,8],[8,2]]

    X, _ = make_blobs(n_samples = 5000, centers = clusters,
    								cluster_std = 0.60)

    # # After training the model, We store the
    # # coordinates for the cluster centers
    ms = MeanShift(bandwidth=2)
    ms.fit(X)
    cluster_centers = ms.centers

    # Finally We plot the data points
    # and centroids in a 3D graph.

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.scatter(X[:, 0], X[:, 1], X[:,2],marker ='o',c=ms.labels)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker ='x', color ='red',
    		s = 300, linewidth = 5, zorder = 1)

    plt.show()

if __name__ == "__main__":
    main()