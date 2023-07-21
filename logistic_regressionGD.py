import numpy as np

class LogisticRegressionGD:
    def __init__(self, learning_rate=.05, number_of_iterations=50, random_state=1) -> None:
        self.learning_rate=learning_rate
        self.number_of_iterations=number_of_iterations
        self.random_state=random_state
    
    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=.0, scale=0.01, size=1 + x.shape[1])
        self.cost = []

        for _ in range(self.number_of_iterations):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = y - output
            self.w[1:] += self.learning_rate * x.T.dot(errors)
            self.w[0] += self.learning_rate * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost.append(cost)
        
        return self
    
    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]
    
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)