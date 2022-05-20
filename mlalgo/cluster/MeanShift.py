import numpy as np
# from .neighbours import kernel_density_estimation


from math import sqrt, pi
def kernel_density_estimation(X:np.ndarray, kernel:str="gaussian", bandwidth: float=1):
    if kernel != "gaussian":
        raise NotImplementedError(f"'{kernel}' kernel hasn't been implemented yet. Use 'gaussian' instead")
    
    return 1/(bandwidth*sqrt(2*pi)) * np.exp(-0.5 * (X/bandwidth)**2)


class MeanShift:
    def __init__(self, bandwidth: float, tolerance: float = 1e-2,verbose:bool = False) -> None:
        self.bandwidth = bandwidth
        self.tolerance = tolerance
        self.verbose = verbose

    def __check_params(self):
        if self.tolerance < 0:
            raise ValueError("tolerance must be greater than 0")
        if self.bandwidth < 0:
            raise ValueError("bandwidth must be greater than 0")

    def fit(self, X: np.ndarray):
        self.__check_params()
        self.labels = np.ndarray((X.shape[0],))
        self.centers = np.ndarray((0,X.shape[1]))
        for i,x in enumerate(X):
            center = x.copy()
            old_center = np.inf
            while not np.allclose(old_center,center,rtol=self.tolerance,atol=0):
                old_center = center.copy()
                neighbours = self.__get_neighbours(X, center)
                kde = kernel_density_estimation(neighbours - center)
                center = np.sum(kde * neighbours,axis=0)/np.sum(kde,axis=0)
            
            existing = False
            for j,c in enumerate(self.centers):
                if np.allclose(c, center,rtol=0.1,atol=1):
                    self.labels[i] = j
                    existing = True
            
            if not existing:
                self.centers = np.append(self.centers, [center],axis=0)
                self.labels[i] = self.centers.shape[0] - 1

        return self

    def predict(self, X: np.ndarray)-> np.ndarray:
        distances = np.ndarray((self.centers,X.shape[0]))
        for i,c in enumerate(self.centers):
            distances[i] = np.linalg.norm(X - c, ord=2, axis=1)

        return np.argmin(distances,axis=0)

    def __get_neighbours(self, X, center):
        distances = np.linalg.norm(X - center, ord=2, axis=1)
        return X[distances <= self.bandwidth]
