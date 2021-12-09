import numpy as np

class LinearRegression:
    def __init__(self, iterations=100000, alpha=0.01):
        self.iterations = iterations
        self.alpha = alpha
        self._norm_coef = 1
    
    def fit(self, X, Y):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("features and target are incompatible") 
        
        feature_vect = np.insert(X.copy(), 0, 1, axis=1)
        
        self.weights = np.ones(feature_vect.shape[1])
        cost = 0
        for _ in range(self.iterations):
            error = feature_vect.dot(self.weights) - Y
            self.weights = self.__gradient_descent(self.weights, feature_vect, error)
            prev = cost
            cost = self.__compute_cost(error)
            if round(cost,5) == round(prev,5):
                return self
        return self
    
    def predict(self, X):
        feature_vect = np.insert(X.copy(), 0, 1, axis=1)
        return np.array([self.weights.transpose().dot(example) for example in feature_vect])

    def __compute_cost(self, error):
        return 0.5 * np.sum(error ** 2)

    def __gradient_descent(self, weights, features, error):
        return weights - self.alpha / error.shape[0] * features.T.dot(error) 

    def score(self, X, Y):
        return 1 - np.sum((self.predict(X) - Y)**2) / np.sum((Y - np.mean(Y))**2)