import numpy as np

# D. Cashon
# 2019 05 06
# numpy implementation of single hidden layer FC net

class NumpyNet:
    def __init__(self, input_num, hidden_num, output_num):
        # initialize weights, biases, and activation functions
        # Xavier-like initialization, found to help with gradient 
        # stability
        self.w_1 = np.sqrt(2 / hidden_num) * np.random.standard_normal(size=(hidden_num, input_num))
        self.b_1 = np.random.normal() * np.ones(hidden_num)
        self.w_2 = np.sqrt(2 / output_num) * np.random.standard_normal(size=(output_num, hidden_num))
        self.b_2 = np.random.normal() * np.ones(output_num)
        self.relu = lambda x: np.maximum(x, 0)
        self.drelu = lambda y: np.piecewise(y, [y >= 0, y < 0], [1, 0])
        self.softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
        self.softmax_vec = lambda x: np.exp(x) / np.sum(np.exp(x))
        # dimensions
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        # adam parameters
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1E-8
        self.t = 1

    def train(self, batch_x, batch_y, lr, method='standard'):
        """
        Computes forward pass and performs weights update
        Parameters:
            batch_x: feature batch
            batch_y: label batch
        """
        num_samples = batch_x.shape[0]
        w_1_i, w_1_j = self.w_1.shape
        w_2_i, w_2_j = self.w_2.shape

        self.grad_w_1 = np.zeros(self.w_1.shape)
        self.grad_w_2 = np.zeros(self.w_2.shape)
        self.grad_b_1 = np.zeros(self.hidden_num)
        self.grad_b_2 = np.zeros(self.output_num)
        self.grad_y_1 = np.zeros(w_1_i)
        self.grad_y_2 = np.zeros(w_2_i)
        self.grad_x_1 = np.zeros(self.hidden_num)
        if method == 'adam':
            # we need storage for moving averages
            self.vdw_1 = np.zeros(self.w_1.shape)
            self.sdw_1 = np.zeros(self.w_1.shape)
            self.vdb_1 = np.zeros(self.b_1.shape)
            self.sdb_1 = np.zeros(self.b_1.shape)
            self.vdw_2 = np.zeros(self.w_2.shape)
            self.sdw_2 = np.zeros(self.w_2.shape)
            self.vdb_2 = np.zeros(self.b_2.shape)
            self.sdb_2 = np.zeros(self.b_2.shape)
        
        # relatively naive elementwise implementation
        # still some slow for loops
        # but it works
        # batch gradient, ADAM
        for i in range(num_samples):
            f_vec = batch_x[i]
            labels_vec = batch_y[i]
            # hidden layer
            self.y_1 = self.w_1 @ f_vec + self.b_1
            self.x_1 = self.relu(self.y_1)
            # output layer
            self.y_2 = self.w_2 @ self.x_1 + self.b_2
            self.x_2 = self.softmax_vec(self.y_2)
            # w_2 update
            # partial E / partial y_2_i
            # vectorized
            self.grad_y_2 = (self.x_2 * ( 1- self.x_2) * 
                            (-1 * labels_vec / self.x_2))
            # for loop (slow)
            #for i in range(w_2_i):
            #    self.grad_y_2[i] = (self.x_2[i] * (1 - self.x_2[i]) * 
            #        (-1 * labels_vec[i] / self.x_2[i]))
            # partial E / partial w_2_ij
            # vectorized version
            self.grad_w_2 += (np.reshape(self.grad_y_2, (self.output_num, 1)) @ np.reshape(self.x_1, (1, self.hidden_num)))
            self.grad_b_2 += self.grad_y_2
            #for loop version (slow)
            # for i in range(w_2_i):
            #     for j in range(w_2_j):
            #         self.grad_w_2[i, j] += self.grad_y_2[i] * self.x_1[j]

            # partial E / partial x_k_1
            for k in range(self.hidden_num):
                self.grad_x_1[k] = np.sum(self.w_2[:,k] * self.grad_y_2)
            # partial E / partial y_1
            # vectorized
            self.grad_y_1 = self.drelu(self.y_1) * self.grad_x_1
            # for loop (slow)
            #for i in range(self.hidden_num):
            #    self.grad_y_1[i] = self.drelu(self.y_1[i]) * self.grad_x_1[i]
            # partial E / partial w_1_ij
            # vectorized
            self.grad_w_1 += (np.reshape(self.grad_y_1, (self.hidden_num, 1)) @ np.reshape(f_vec, (1, self.input_num)))
            self.grad_b_1 += self.grad_y_1
            # for loop version (slow)
            # for i in range(w_1_i):
            #     for j in range(w_1_j):
            #         self.grad_w_1[i, j] += f_vec[j] * self.grad_y_1[i]
        # weights update
        if method == 'standard':
            # perform usual batch gd update
            self.w_1 -= lr * self.grad_w_1 / num_samples
            self.w_2 -= lr * self.grad_w_2 / num_samples
            self.b_1 -= lr * self.grad_b_1 / num_samples
            self.b_2 -= lr * self.grad_b_2 / num_samples
        elif method == 'adam':
            # compute update
            # there is no doubt a better way to do this
            # for hidden weight bias
            self.vdw1_current = self.beta_1 * self.vdw_1 + (1-self.beta_1) * (self.grad_w_1 / num_samples)
            np.copyto(self.vdw_1, self.vdw1_current)
            self.vdb1_current = self.beta_1 * self.vdb_1 + (1-self.beta_1) * (self.grad_b_1 / num_samples)
            np.copyto(self.vdb_1, self.vdb1_current)
            self.sdw1_current = self.beta_2 * self.sdw_1 + (1 - self.beta_2) * (self.grad_w_1 / num_samples) ** 2
            np.copyto(self.sdw_1, self.sdw1_current)
            self.sdb1_current = self.beta_2 * self.sdb_1 + (1 - self.beta_2) * (self.grad_b_1 / num_samples) ** 2
            np.copyto(self.sdb_1, self.sdb1_current)
            # for output weights biases
            self.vdw2_current = self.beta_1 * self.vdw_2 + (1-self.beta_1) * (self.grad_w_2 / num_samples)
            np.copyto(self.vdw_2, self.vdw2_current)
            self.vdb2_current = self.beta_1 * self.vdb_2 + (1-self.beta_1) * (self.grad_b_2 / num_samples)
            np.copyto(self.vdb_2, self.vdb2_current)
            self.sdw2_current = self.beta_2 * self.sdw_2 + (1 - self.beta_2) * (self.grad_w_2 / num_samples) ** 2
            np.copyto(self.sdw_2, self.sdw2_current)
            self.sdb2_current = self.beta_2 * self.sdb_2 + (1 - self.beta_2) * (self.grad_b_2 / num_samples) ** 2
            np.copyto(self.sdb_2, self.sdb2_current)
            # compute corrections
            self.vdw1_current = self.vdw1_current / (1 - self.beta_1 ** self.t)
            self.vdw2_current = self.vdw2_current / (1 - self.beta_1 ** self.t)
            self.sdw1_current = self.sdw1_current / (1 - self.beta_2 ** self.t)
            self.sdw2_current = self.sdw2_current / (1 - self.beta_2 ** self.t)
            self.vdb1_current = self.vdb1_current / (1 - self.beta_1 ** self.t)
            self.vdb2_current = self.vdb2_current / (1 - self.beta_1 ** self.t)
            self.sdb1_current = self.sdb1_current / (1 - self.beta_2 ** self.t)
            self.sdb2_current = self.sdb2_current / (1 - self.beta_2 ** self.t)
            # update weights
            self.w_1 -= lr * (self.vdw1_current / (np.sqrt(self.sdw1_current) + self.epsilon))
            self.w_2 -= lr * (self.vdw2_current / (np.sqrt(self.sdw2_current) + self.epsilon)) 
            self.b_1 -= lr * (self.vdb1_current / (np.sqrt(self.sdb1_current) + self.epsilon)) 
            self.b_2 -= lr * (self.vdb2_current / (np.sqrt(self.sdb2_current) + self.epsilon))

    def forward_pass(self, batch_x):
        """
        Forward pass, outputting logits. Used to predict
        Parameters:
            batch_x: numpy array of n samples d features
        Returns:
            logits: numpy array of output logits post-softmax
        """
        self.logits = self.softmax(self.relu(batch_x @ self.w_1.T + self.b_1) @ self.w_2.T + self.b_2)
        return self.logits

    def get_error(self, batch_x, batch_y):
        # unused
        return 0
    
    def grad_descent(self):
        # unused
        return 0
    
    def back_propagate(self):
        # unused
        return 0

    def reset_weights(self):
        # unused
        return 0

