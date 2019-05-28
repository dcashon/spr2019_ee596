import tensorflow as tf
import numpy as np

"""
TO: Define your char rnn model here

You will define two functions inside the class object:

1) __init__(self, args_1, args_2, ... ,args_n):

    The initialization function receives all hyperparameters as arguments.

    Some necessary arguments will be: batch size, sequence_length, vocabulary size (number of unique characters), rnn size,
    number of layers, whether use dropout, learning rate, use embedding or one hot encoding,
    and whether in training or testing,etc.

    You will also define the tensorflow operations here. (placeholder, rnn model, loss function, training operation, etc.)


2) sample(self, sess, char, vocab, n, starting_string):
    
    Once you finish training, you will use this function to generate new text

    args:
        sess: tensorflow session
        char: a tuple that contains all unique characters appeared in the text data
        vocab: the dictionary that contains the pair of unique character and its assoicated integer label.
        n: a integer that indicates how many characters you want to generate
        starting string: a string that is the initial part of your new text. ex: 'The '

    return:
        a string that contains the genereated text

"""
class Model():
    def __init__(self, num_neurons, seq_length, vocab_size, num_layers=2, lr=0.001, use_dropout=True, use_gpu=True):
        """
        Initializes a character to character RNN model with multilayer LSTM architecture
        Architecture:
            User Parameters:
                -num_neurons:   neurons per cell
                -num_layers:    number of layers
                -use_dropout:   whether dropout is used, default=True
                -learning_rate: learning rate, default=0.001
            Defaults:
                -optimizer: adam
                -loss: mean cross entropy
                -output dropout probability: 0.2

        Inputs are assumed to be one-hot encoded characters of shape [batch_size, seq_length, len(vocab)]
        (Data is one-hot encoded in preprocessing, see text_utils.py)
        """
        # check
        assert (num_layers > 1), "Must be multilayer"
        assert (seq_length > 1), "Cannot use short seq"

        tf.reset_default_graph()

        # model parameters
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.num_steps = seq_length
        self.num_inputs = vocab_size # one-hot encode size will be size of unique vocab
        self.lr = lr
        self.use_gpu = use_gpu
        #self.num_outputs = 1 # 1 predicted next character
        self.vocab_size = vocab_size # number of unique characters

        # construct the graph
        self.X = tf.placeholder(tf.float32, shape=[None, self.num_steps, self.num_inputs])
        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_steps, self.num_inputs])

        # RNN
        if self.use_gpu:
            base_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
        else:
            base_cell = tf.contrib.rnn.BasicLSTMCell
        
        self.layers = [base_cell(num_units=self.num_neurons) for i in range(self.num_layers)]
        if use_dropout is True:
            self.drops = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8) for cell in self.layers] #hardcoded...
            self.multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.drops)
        else:
            self.multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.layers)
            
        self.outputs, self.states = tf.nn.dynamic_rnn(self.multi_layer_cell, self.X, dtype=tf.float32)
        # we use a many to many
        # output has shape [batch_size, n_steps, num_neurons]
        # reshape
        self.output_reshape = tf.reshape(self.outputs, [-1, self.num_neurons])
        self.stacked_outputs = tf.layers.dense(self.output_reshape, self.vocab_size)
        self.logits = tf.reshape(self.stacked_outputs, [-1, self.num_steps, self.vocab_size])
        # last output state has a fully connected layer to softmax for character prediction
        # LSTM cell has two outputs, so we need the hidden state of the "top" layer
        #self.logits = tf.layers.dense(self.states[self.num_layers-1][0], self.vocab_size) # one-hot

        # loss and optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logits))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr) # default optimizer
        self.trainer = self.opt.minimize(self.loss)

    def train(self, batch_x, batch_y):
        return 0

    def sample(self):
        # I implemented this in the Jupyter notebook instead
        return 0
