import tensorflow as tf



class StockModel():
    def __init__(self, num_neurons):
        """
        Initializes simple RNN for stock prediction
        """
        tf.reset_default_graph()
