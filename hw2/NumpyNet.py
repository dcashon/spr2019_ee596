import numpy as np



class NumpyNet:
    def __init__(self, input_num, hidden_num, output_num):
        # initialize weights, biases, and activation functions
        self.w_1 = np.random.standard_normal(size=(input_num, hidden_num))
        self.b1 = np.random.normal()
        self.w_out = np.random.standard_normal(size=(hidden_num, output_num))
        self.b_out = np.random.normal()
        self.relu = lambda x: np.maximum(x, 0)
        self.softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]

    def forward_pass(self, batch_x, batch_y):
        # forward pass
        # TODO
        return 0
    
    def grad_descent(self):
        # gradient descent
        # TODO
        return 0
    
    def back_propagate(self):
        # backprop
        # TODO
        return 0

    def reset_weights(self):
        # reset all weights
        # TODO
        return 0

