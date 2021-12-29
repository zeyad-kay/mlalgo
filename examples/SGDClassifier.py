import numpy as np
from sklearn.linear_model import SGDClassifier as sklModel
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlalgo import SGDClassifier

def main():
    iris = load_iris()
    X = iris.data
    Y = iris.target
    
    lbls = ['setosa', 'versicolor', 'verginica']
    lbls_combinations = [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
    cols_combinations = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    for lc in lbls_combinations:
        print(f"{lbls[lc[0]]} vs {lbls[lc[1]]}:")
        idx = np.where((Y==lc[0]) | (Y==lc[1]))
        xtrain,xtest,ytrain,ytest = train_test_split(X[idx],Y[idx],test_size=0.3,random_state=0)
        for c in cols_combinations: 
            ytrn = ytrain.copy()
            xtrn = xtrain[:,c].copy()
            ytst = ytest.copy()
            xtst = xtest[:,c].copy()
            ytrn[ytrn==lc[0]] = -1
            ytrn[ytrn==lc[1]] = 1
            ytst[ytst==lc[0]] = -1
            ytst[ytst==lc[1]] = 1
            skl,custom = get_classifiers(xtrn, ytrn)
            plot_models(custom, skl, xtrn, ytrn, lbls[lc[0]],lbls[lc[1]], xlabel=cols[c[0]], ylabel=cols[c[1]])
            print(f"\tx1={cols[c[0]]}, x2={cols[c[1]]}:")
            print(f"\t\tsklearn accuracy: {skl.score(xtst, ytst)}")
            print(f"\t\tmy accuracy: {custom.score(xtst, ytst)}")

def get_classifiers(x, y):
    # implementation
    clf = SGDClassifier()
    clf.fit(x,y)
    # sklearn classifier
    skclf = sklModel(loss="hinge", penalty="l2", random_state=1)
    skclf.fit(x,y)

    return skclf, clf

def plot_models(clf, skclf, x ,y, lbl0, lbl1 ,xlabel, ylabel):
    plt.clf()
    plt.plot(np.array([0,1,2,3,4,5,6,7,8]), clf.intercept / - clf.coef[1] + clf.coef[0] / - clf.coef[1] * np.array([0,1,2,3,4,5,6,7,8]))
    plt.plot(np.array([0,1,2,3,4,5,6,7,8]), skclf.intercept_ / - skclf.coef_[0][1] + skclf.coef_[0][0] / - skclf.coef_[0][1] * np.array([0,1,2,3,4,5,6,7,8]))

    plt.scatter(x[y==-1][:,0], x[y==-1][:,1] ,c="red", label=lbl0)
    plt.scatter(x[y==1][:,0], x[y==1][:,1] ,c="green", label=lbl1)
    plt.legend(["My Model","sklearn", lbl0, lbl1])
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    plt.show()

if __name__ == "__main__":
    main()