from math import log
import numpy as np
from collections import deque

class Node:
    def __init__(self,criterion: str,labels: dict, impurity: float, feature: int|None=None,threshold=None,left=None,right=None) -> None:
        self.left = left
        self.right = right
        self.impurity = impurity
        self.feature = feature
        self.threshold = threshold
        self.labels = labels
        self.criterion = criterion
    
    def isLeaf(self) -> bool:
        if self.left and self.right:
            return False
        else:
            return True
    
    def __str__(self) -> str:
        if self.isLeaf():
            return f"{self.criterion}={self.impurity}, Labels={self.labels}"
        else:
            return f"X[{self.feature}]={self.threshold}, {self.criterion}={self.impurity}, Labels={self.labels}"

class DecisionTreeClassifier:
    def __init__(self, max_depth:int|None=None, min_samples:int=1, criterion:str="gini"):
        self._root = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.criterion = criterion

    def fit(self, X, Y):
        self.__check_params()
        return self.__build_tree(X, Y)
    
    def predict(self, X):
        if self._root:
            return np.array([self.__traverse(self._root, x) for x in X])
        else:
            raise Exception("Tree has no root. Fit the model first.")

    def score(self, X, Y):
        ps = self.predict(X)
        return ps[ps == Y].shape[0] / ps.shape[0]
    
    def __check_params(self):
        if not self.max_depth or self.max_depth < 0:
            raise ValueError(f"max_depth must be greater than zero.")

        if self.min_samples < 0:
            raise ValueError(f"min_samples must be greater than or equal zero.")

        if self.criterion not in ["gini", "entropy"]:
            raise ValueError(f"{self.criterion} is not a valid criterion.")

    def __traverse(self, node: Node, x):
        if node.isLeaf():
            return max(node.labels,key=node.labels.get)

        if x[node.feature] == node.threshold:
            return self.__traverse(node.left,x)
        else:
            return self.__traverse(node.right,x)

    def __ylabels(self, Y):
        targets, targets_count = np.unique(Y, return_index=False, return_counts=True)
        return dict(zip(targets,targets_count))
    
    def __build_tree(self, X, Y):
              
        # pick the feature with min impurity
        feature, threshold = self.__optimal_split(X, Y)
        
        lbls = self.__ylabels(Y)
        self._root = Node(self.criterion, labels=lbls, feature=feature, impurity=self.__impurity(lbls), threshold=threshold)
        
        lbls = self.__ylabels(Y[X[:,feature] == threshold])
        self._root.left = Node(self.criterion, labels=lbls, impurity=self.__impurity(lbls))
        
        lbls = self.__ylabels(Y[X[:,feature] != threshold])
        self._root.right = Node(self.criterion, labels=lbls, impurity=self.__impurity(lbls))

        q = deque()
        q.append({ "node": self._root.left, "X": X[X[:,feature] == threshold], "Y": Y[X[:,feature] == threshold] })
        q.append({ "node": self._root.right, "X": X[X[:,feature] != threshold], "Y": Y[X[:,feature] != threshold] })
        while len(q) != 0:
            n = q.pop()
            if self.__node_depth(n["node"]) == self.max_depth:
                continue        
            if min(n["node"].labels.values()) <= self.min_samples or len(n["node"].labels) <= 1:
                continue

            n["node"].feature, n["node"].threshold = self.__optimal_split(n["X"], n["Y"])
            
            # sometimes the minimum feature has 1 label only
            # this results in the same split 
            if n["Y"][n["X"][:,n["node"].feature] == n["node"].threshold].shape[0] == n["Y"].shape[0]:
                n["node"].feature = n["node"].threshold = None
                continue

            lbls = self.__ylabels(n["Y"][n["X"][:,n["node"].feature] == n["node"].threshold])
            if lbls:
                n["node"].left = Node(self.criterion, labels=lbls, impurity=self.__impurity(lbls))
                q.append({ "node": n["node"].left, "X": n["X"][n["X"][:,n["node"].feature] == n["node"].threshold], "Y": n["Y"][n["X"][:,n["node"].feature] == n["node"].threshold] })
            
            lbls = self.__ylabels(n["Y"][n["X"][:,n["node"].feature] != n["node"].threshold])
            if lbls:
                n["node"].right = Node(self.criterion, labels=lbls, impurity=self.__impurity(lbls))
                q.append({ "node": n["node"].right, "X": n["X"][n["X"][:,n["node"].feature] != n["node"].threshold], "Y": n["Y"][n["X"][:,n["node"].feature] != n["node"].threshold] })

        return self

    def depth(self):
        q = deque()
        q.append(self._root)
        height = 0 
        while(True):
            nodeCount = len(q)
            if nodeCount == 0 :
                return height - 1 

            height += 1 
            while(nodeCount > 0):
                node = q.popleft()
                if node.left is not None:
                    q.append(node.left)
                if node.right is not None:
                    q.append(node.right)
                nodeCount -= 1
    
    def __node_depth(self, target_node):
        q = deque()
        q.append((self._root,0))
        while len(q) != 0:
            node,d = q.pop()
            if node is target_node:
                return d
            else:
                if node.left:
                    q.append((node.left,d+1))
                if node.right:
                    q.append((node.right,d+1))
        return -1
    
    def __impurity(self, labels):
        total = sum(labels.values())
        impurity = 0
        for v in labels.values():
            if self.criterion == "gini":
                impurity -= (v/total)**2
            elif self.criterion == "entropy":
                impurity -= v/total * log(v/total, 2)
        if self.criterion == "gini":
            impurity += 1
        return impurity

    def __optimal_split(self, X, Y):
        features_cost = []
        features_threshold = []
        for feature in range(X.shape[1]):
            labels = np.unique(X[:,feature], return_index=False, return_counts=False)
            labels_costs = []
            for lbl in labels:
                left_targets, left_targets_count = np.unique(Y[X[:,feature] == lbl], return_index=False, return_counts=True)
                left_cost = self.__impurity(dict(zip(left_targets, left_targets_count)))
                right_targets, right_targets_count = np.unique(Y[X[:,feature] != lbl], return_index=False, return_counts=True)
                right_cost = self.__impurity(dict(zip(right_targets, right_targets_count)))

                total_cost = Y[X[:,feature] == lbl].shape[0] / Y.shape[0] * left_cost + Y[X[:,feature] != lbl].shape[0] / Y.shape[0] * right_cost
                labels_costs.append(total_cost)
            
            idx = labels_costs.index(min(labels_costs))
            features_threshold.append(labels[idx])            
            features_cost.append(labels_costs[idx])
        
        idx = features_cost.index(min(features_cost))
        return idx, features_threshold[idx]