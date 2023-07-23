import numpy as np

class KNN:
    def __init__(self, k, features, targets):
        self.k = k
        self.features = features
        self.targets = targets

    def predict(self, x):
        distances = self.compute_distances(x)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_targets = self.targets[k_indices]
        return self.majority(k_nearest_targets)

    def majority(self, predictions):
        return np.bincount(predictions).argmax()
    
    def compute_distances(self, x):
        distances = []
        for i in range(len(self.features)):
            distances.append(self.distance(x, self.features[i]))
        return np.array(distances)
    
    def distance(self, x1, x2):
        return np.linalg.norm(x1 - x2, ord=2)