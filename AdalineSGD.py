import numpy as np

class AdalineSGD:
    def __init__(self, learning_rate=0.01, number_of_iterations=30, random_state=1, shuffle=True) -> None:
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.random_state = random_state
        self.shuffle = shuffle

    def fit(self, x, y):
        self._initialize_weights(x.shape[1])
        self.cost = []
        for _ in range(self.number_of_iterations):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost.append(avg_cost)
        return self
    
    def _shuffle(self, x, y):
        np.random.shuffle(x)
        np.random.shuffle(y)
        return x, y

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ =self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized=True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[1:] += self.learning_rate * xi.dot(error)
        self.w_[0] += self.learning_rate * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def activation(self, x):
        return x
    
    def predict(self, x):
        return np.where(self.activation(self.net_input(x)) >= 0, 1, -1)