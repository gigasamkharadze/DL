import numpy as np

class AdalineGD:
    def __init__(self, learning_rate=0.01, number_of_iterations=30, random_state=1) -> None:
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.random_state = random_state

    def fit(self, x, y):
        """_summary_

        Args:
            x ( np-array ): training vectors
            y ( np-array ): target values
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.01, size= 1+x.shape[1])
        self.cost = []

        for _ in range(self.number_of_iterations):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.learning_rate * x.T.dot(errors)
            self.w_[0] += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2
            self.cost.append(cost)
        
        return self
    
    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def activation(self, x):
        return x
    
    def predict(self, x):
        return np.where(self.activation(self.net_input(x))>= 0, 1, -1)