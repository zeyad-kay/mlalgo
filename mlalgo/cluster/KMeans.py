import numpy as np

# TODO: Add kmeans++
class KMeans():
    def __init__(self, K:int=2, n_init:int=10, initial_centers:np.ndarray|None=None ,max_iter:int =300, tolerance:float=1e-4, verbose:bool= False) -> None:
        self.K = K
        self.n_init = n_init
        self.max_iter = max_iter
        self.verbose = verbose
        self.centers = None
        self.initial_centers = initial_centers
        self.tolerance = tolerance

    def __check_params(self):
        if self.K < 2:
            raise ValueError(f"K must be greater than or equal 2")
        if self.n_init < 1:
            raise ValueError(f"init must be greater than or equal 1")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be greater than or equal 1")
        if self.initial_centers is not None and type(self.initial_centers) != np.ndarray:
            raise ValueError(f"initial_centers must be None or ndarray")
            
    
    def fit(self, X: np.ndarray):
        self.__check_params()
        
        best_labels = np.ndarray((X.shape[0],))
        best_centers = np.ndarray((self.K, X.shape[1]))
        best_inertia = np.inf
        for i in range(self.n_init):
            centers, inertia, labels = self.__kmeans(X)
            if inertia < best_inertia:
                best_centers = centers
                best_inertia = inertia
                best_labels = labels
        
        self.labels = best_labels
        self.centers = best_centers
        self.inertia = best_inertia
        return self

    def __kmeans(self, X:np.ndarray):
        centers, inertia = self.__init_centers(X)
        labels = np.ndarray((X.shape[0],))
        for i in range(self.max_iter):

            old_centers, old_inertia, old_labels = centers.copy(), inertia, labels.copy()

            centers, inertia, labels = self.__update(X, centers)

            if np.array_equal(labels, old_labels) and np.allclose(old_centers,centers,rtol=self.tolerance,atol=0):
                if self.verbose:
                    print(f"converged at iteration #{i+1}")
                break

            if self.verbose:
                print(f"iteration #{i+1}:")
                [print(f"\tcentroid {j+1}: {k}") for j,k in enumerate(centers)]

        return centers, inertia, labels

    def predict(self, X: np.ndarray)-> np.ndarray:
        if self.centers is None:
            raise Exception("Fit the model first.")

        distances = np.ndarray((self.K,X.shape[0]))
        for i,c in enumerate(self.centers):
            distances[i] = np.linalg.norm(X - c, ord=2, axis=1)

        return np.argmin(distances,axis=0)

    def score(self, X):
        if self.centers is None:
            raise Exception("Fit the model first.")

        labels = self.predict(X)
        score = 0
        for k in range(self.K):
            x = X[np.where(labels==k)[0]]
            if x.shape[0]:
                score -= (np.linalg.norm(x - self.centers[k], ord=2, axis=1)**2).sum()
        return score
    
    def __update(self, X, centers):
        distances = np.ndarray((self.K,X.shape[0]))
        for i,c in enumerate(centers):
            distances[i] = np.linalg.norm(X - c, ord=2, axis=1)

        labels = np.argmin(distances,axis=0)
        inertia = 0
        for k in range(self.K):
            x = X[np.where(labels==k)[0]]
            if x.shape[0]:
                centers[k] = x.mean(axis=0)
                inertia += (np.linalg.norm(x - centers[k], ord=2, axis=1)**2).sum()
        return centers, inertia, labels

    def __init_centers(self, X):
        # check if user supplied initial centers else
        # initialize random centers from the observations
        if self.initial_centers and self.initial_centers.shape != (self.K, X.shape[1]):
            raise ValueError(f"Expected shape({self.K}, {X.shape[1]}), found {self.initial_centers.shape}") 
        
        inertia = np.inf
        centers = None
        if self.initial_centers:
            centers = self.initial_centers
        else:
            centers = np.array([X[np.random.randint(0,X.shape[0])] for _ in range(self.K)])

        if self.verbose:
            [print(f"initial centroid {i+1}: {k}") for i,k in enumerate(centers)]

        return centers, inertia