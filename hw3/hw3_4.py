
# coding: utf-8

# In[1]:


#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))


# In[1]:


import tensorflow as tf
import timeit
import os
from six.moves import cPickle
from text_utils import TextLoader
from tensorflow.contrib import rnn
from char_rnn_model import Model
import numpy as np


# # Define directories, hyperparameter

# In[2]:


# data locations, etc
data_dir = '.'

# training
num_epochs = 15
batch_size = 512

# network architecture
num_neurons = 256
seq_length = 50
vocab_size = 67
num_layers = 2 
dropout = False
learning_rate = 0.001


# # Load data using TextLoader object

# In[3]:


loader = TextLoader('.', 20, seq_length)
#loader.preprocess_data()


# # Create your model object

# In[4]:


my_model2 = Model(num_neurons, seq_length, vocab_size, num_layers=3, lr=0.001, use_dropout=True, use_gpu=True)


# ## Save

# In[5]:


saver = tf.train.Saver()


# # Training

# In[7]:
# load validation
x_val = np.load("./s_val.npy")
x_val_data = x_val[:512, :-1, :]
x_val_labels = x_val[:512, -1, :]

init = tf.global_variables_initializer()
i = 0
with tf.Session() as sess:
    sess.run(init)
    for epochs in range(num_epochs):
        print("Training for Epoch Number: \t" + str(epochs))
        for num in range(5):
            print("Loading Data...")
            x_train = loader.load_train_data(num)
            print("Batching Data on:\t " + str(num))
            batcher = loader.training_batch(batch_size)
            print("Beginning Training:")
            for data, labels in batcher:
                i+=1
                f_dict = {my_model2.X:data, my_model2.Y:labels}
                sess.run(my_model2.trainer, feed_dict=f_dict)
                if i % 50 ==0:
                    c_loss = sess.run(my_model2.loss, feed_dict=f_dict)
                    print(c_loss)
        save_path = saver.save(sess, "./s_checkpoints/model" + str(epochs) + str(num+1) + ".ckpt")
        print("Model saved in path: %s" % save_path)
    val_loss = sess.run(my_model2.loss, {my_model2.X:x_val_data, my_model2.Y:x_val_labels})
    print("Validation Loss is: \t" + str(val_loss))






