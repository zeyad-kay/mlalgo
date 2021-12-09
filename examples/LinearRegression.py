from mlalgo import LinearRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

def normalize(features):
    cpy = np.ndarray(features.shape)
    for i in range(features.shape[1]):
        cpy[:,i] = (features[:,i] - features[:,i].min()) / (features[:,i].max() - features[:,i].min())
    return cpy

def read_dat(filename):
    data = []
    with open(filename) as f:
        [data.append(list(map(lambda x: float(x), line.rstrip("\n").split(",")))) for line in f.readlines()]
    return np.array(data)


def main():
    univariate = read_dat("data/univariateData.dat")
    univariate = normalize(univariate)
    xtrain,xtest,ytrain,ytest = train_test_split(univariate[:,:-1],univariate[:,-1],test_size=0.3,shuffle=False)    

    cmodel = LinearRegression()
    cmodel.fit(xtrain,ytrain)
    print("Training Univariate Data:")
    print(f"\t mlalgo accuracy: {cmodel.score(xtest,ytest)}")

    skmodel = skLinearRegression()
    skmodel.fit(xtrain,ytrain)    
    print(f"\t sklearn accuracy: {skmodel.score(xtest,ytest)}")
    
    multivariate_data = read_dat("data/multivariateData.dat")
    multivariate_data = normalize(multivariate_data)
    xtrain,xtest,ytrain,ytest = train_test_split(multivariate_data[:,:-1],multivariate_data[:,-1],test_size=0.3,shuffle=False)    

    cmodel = LinearRegression()
    cmodel.fit(xtrain,ytrain)
    print("Training Multivariate Data:")
    print(f"\t mlalgo accuracy: {cmodel.score(xtest,ytest)}")

    skmodel = skLinearRegression()
    skmodel.fit(xtrain,ytrain)    
    print(f"\t sklearn accuracy: {skmodel.score(xtest,ytest)}")

if __name__ == "__main__":
    main()