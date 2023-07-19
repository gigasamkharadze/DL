import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, number_of_iterations=30, random_state=1) -> None:
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.random_state = random_state

    def fit(self, x, y):
        """
        _summary_
        
        Args:
            x ( np-array ): training vectors
            y ( np-array ): target values
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0, scale=.01, size= 1 + x.shape[1])
        self.errors = []

        for _ in range(self.number_of_iterations):
            errors = 0
            for xi, target in zip(x, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors.append(errors)
        
        return self
    
    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def predict(self, x):
        return np.where(self.net_input(x) >= 0, 1, -1)
