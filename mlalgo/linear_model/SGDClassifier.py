import numpy as np

class SGDClassifier:
    def __init__(self, max_iterations:int=1000, alpha:float=0.001, eta:float=0.001, tolerance:float=0.001):
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.eta = eta
        self.tolerance = tolerance

    def fit(self, X:np.ndarray, Y:np.ndarray):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have same dimensions.") 
        self.__sgd(X, Y)
        return self

    def _check_params(self):
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be greater than 0.") 
        if self.eta < 0 or self.eta > 1:
            raise ValueError("learning rate must be between 0 and 1.") 
        if self.tolerance < 0:
            raise ValueError("tolerance must be greater than 0.") 

    def __sgd(self, X:np.ndarray, Y:np.ndarray):
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Y must have shape {(X.shape[0],)}")
        
        self.coef = np.zeros(X.shape[1])
        self.intercept = 0
        i = 0
        while self.loss(X, Y) > self.tolerance and i < self.max_iterations:
            for j in range(Y.shape[0]):
                self.__step(X[j], Y[j])
            i += 1
    
    def __step(self, x, y):
        regularization = self.__regularization_gradient(self.coef)
        w_loss, b_loss = self.__loss_gradient(x, y, self.coef, self.intercept)
        self.coef = self.coef - self.eta * (w_loss + self.alpha * regularization)
        self.intercept = self.intercept - self.eta * b_loss

    def loss(self, X:np.ndarray, Y:np.ndarray):
        N = Y.shape[0]
        L = 1 - Y * (X.dot(self.coef) + self.intercept)
        L[L < 0] = 0
        return 1 / N * sum(L) + (self.alpha * self.__regularization(self.coef))

    def __regularization_gradient(self, w:np.ndarray):
        return w.sum()

    def __loss_gradient(self, x, y, w, b):
        if max(0, 1 - y * (np.dot(x, w) + b)) == 0:
            return 0,0
        else:
            return - np.dot(y,x), - y

    def __regularization(self, w:np.ndarray):
        return 0.5 * sum(w ** 2)
    
    def predict(self, X:np.ndarray):
        if self.coef is None:
            raise Exception("Fit the model first")
        if X.shape[1] != self.coef.shape[0]:
            raise ValueError(f"X must have shape {(X.shape[0],self.coef.shape[0])}")
        y = X.dot(self.coef) + self.intercept
        y[y > 0] = 1
        y[y < 0] = -1
        return y

    def score(self, X:np.ndarray, Y:np.ndarray):
        ypred = self.predict(X)
        return ypred[ypred == Y].shape[0] / Y.shape[0]
