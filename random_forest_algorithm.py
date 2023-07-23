# implement random foresr algorithm

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, criterion="gini", n_estimators=25, random_state=1):
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(criterion=self.criterion, random_state=self.random_state)
            self.trees.append(tree)

    def fit(self, X, y):
        for tree in self.trees:
            bootstrap_sample = self.bootstrap_sample(X, y)
            tree.fit(*bootstrap_sample)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return self.majority(tree_predictions.T)
    
    def majority(self, predictions):
        return np.array([np.bincount(prediction).argmax() for prediction in predictions])

    def bootstrap_sample(self, X, y):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        return X[indices], y[indices]