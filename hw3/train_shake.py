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
num_epochs = 5
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


loader = TextLoader('.', 20, 50)
#loader.preprocess_data()


# # Create your model object

# In[4]:


char_model = Model(num_neurons, seq_length, vocab_size, num_layers=3, lr=0.001, use_dropout=True, use_gpu=True)


# ## Save

# In[5]:


saver = tf.train.Saver()


# # Training

# In[ ]:


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
                print(data.shape)
                i+=1
                f_dict = {char_model.X:data[:-1], char_model.Y:data[1:]}
                sess.run(char_model.trainer, feed_dict=f_dict)
                #test = sess.run(char_model.outputs, feed_dict=f_dict)
                #print(test.shape)
                if i % 1 ==0:
                    c_loss = sess.run(char_model.loss, feed_dict=f_dict)
                    print(c_loss)
        save_path = saver.save(sess, "./tmp/model_02" + str(epochs) + str(num) + ".ckpt")
        print("Model saved in path: %s" % save_path)






