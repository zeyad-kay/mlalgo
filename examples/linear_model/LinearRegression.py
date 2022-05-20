from mlalgo.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    x = np.random.rand(9,3)
    b = np.random.randint(-10,10)
    w = np.random.randint(-10,10,(3))
    y = b + np.dot(x,w)
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)    

    cmodel = LinearRegression(fit_intercept=False)
    cmodel.fit(xtrain,ytrain)
    print(f"mlalgo accuracy: {round(cmodel.score(xtest,ytest),2)}")

    skmodel = skLinearRegression(fit_intercept=False)
    skmodel.fit(xtrain,ytrain)    
    print(f"sklearn accuracy: {round(skmodel.score(xtest,ytest),2)}")

    print(f"Difference in coefficients: {np.abs(skmodel.coef_ - cmodel.coef).round(2)}")
    print(f"Difference in intercept: {round(skmodel.intercept_ - cmodel.intercept,2)}")

if __name__ == "__main__":
    main()