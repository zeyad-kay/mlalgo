import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept: bool=True):
        self.fit_intercept = fit_intercept

    def fit(self, X:np.ndarray, Y:np.ndarray):
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Y must have shape {(X.shape[0],)}")
        
        copy_X = np.insert(X, 0, 1, axis=1) if self.fit_intercept else X.copy()
        
        copy_XT = np.transpose(copy_X)
        
        w = np.matmul(np.linalg.inv(np.matmul(copy_XT, copy_X)), np.matmul(copy_XT, Y))

        if self.fit_intercept:
            self.intercept = w[0]
            self.coef = w[1:]
        else:
            self.coef = w
            self.intercept = 0
        
        return self
    
    def predict(self, X:np.ndarray):
        if X.shape[1] != self.coef.shape[0]:
            raise ValueError(f"X must have shape {(X.shape[0],self.coef.shape[0])}") 

        return np.dot(X,self.coef) + self.intercept

    def score(self, X:np.ndarray, Y:np.ndarray):
        return 1 - np.sum((self.predict(X) - Y)**2) / np.sum((Y - np.mean(Y))**2)
