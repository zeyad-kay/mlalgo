from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas
from mlalgo import DecisionTreeClassifier
import time

def main():
    data = pandas.read_csv("data/cardio_train.csv",header=0)

    xtrain,xtest,ytrain,ytest = train_test_split(data.iloc[:,1:-1].values,data["cardio"].values,test_size=0.1,shuffle=False)    
    
    mlalgomodel = DecisionTreeClassifier(criterion='entropy',min_samples=2)
    s=time.perf_counter()
    mlalgomodel.fit(xtrain,ytrain)
    t=time.perf_counter() - s
    print(f"mlalgo tree: ")
    print(f"\tdepth: {mlalgomodel.depth()}")
    print(f"\taccuracy: {mlalgomodel.score(xtest,ytest)}")
    print(f"\tFinished in {t} second(s).")

    skl = tree.DecisionTreeClassifier(criterion='entropy')
    s=time.perf_counter()
    skl.fit(xtrain,ytrain)
    t=time.perf_counter() - s
    print(f"sklearn tree: ")
    print(f"\tdepth: {skl.get_depth()}")
    print(f"\taccuracy: {skl.score(xtest,ytest)}")
    print(f"\tFinished in {t} second(s).")

if __name__ == "__main__":
    main()