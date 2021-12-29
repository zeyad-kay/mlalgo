import numpy as np

class SGDClassifier:
    def __init__(self, max_iterations=10000, alpha=0.001, learning_rate=0.001, tolerance=0.001):
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.eta = learning_rate
        self.tolerance = tolerance

    def fit(self, X, Y):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have same dimensions.") 
        self.coef, self.intercept = self.__sgd(X, Y)

    def __sgd(self, X, Y):
        w = np.zeros(X.shape[1])
        b = 0
        for _ in range(self.max_iterations):
            if self.cost(X, Y, w, b) <= self.tolerance:
                break

            for x,y in zip(X,Y):
                regularization = self.__regularization_gradient(w)
                w_loss, b_loss = self.__loss_gradient(x, y, w, b)
                w = w - self.eta * (w_loss + self.alpha * regularization)
                b = b - self.eta * b_loss
        return w, b
    
    def cost(self, X, Y, w, b):
        N = Y.shape[0]
        L = 1 - Y * (X.dot(w) + b)
        L[L < 0] = 0
        return 1 / N * sum(L) + (self.alpha * self.__regularization(w))

    def __regularization_gradient(self, w):
        return sum(w)

    def __loss_gradient(self, x, y, w, b):
        if max(0, 1 - y * (np.dot(x, w) + b)) == 0:
            return 0,0
        else:
            return - np.dot(y,x), - y

    def __regularization(self, w):
        return 0.5 * sum(w ** 2)
    
    def predict(self, X):
        y = X.dot(self.coef) + self.intercept
        y[y > 0] = 1
        y[y < 0] = -1
        return y

    def score(self, X, Y):
        ypred = self.predict(X)
        return ypred[ypred == Y].shape[0] / Y.shape[0]
