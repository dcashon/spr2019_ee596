import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
import re

"""
Implement a class object that should have the following functions:

1) object initialization:
This function should be able to take arguments of data directory, batch size and sequence length.
The initialization should be able to process data, load preprocessed data and create training and
validation mini batches.

2)helper function to preprocess the text data:
This function should be able to do:
    a)read the txt input data using encoding='utf-8'
    b)
    b1)create self.char that is a tuple contains all unique character appeared in the txt input.
    b2)create self.vocab_size that is the number of unique character appeared in the txt input.
    b3)create self.vocab that is a dictionary that the key is every unique character and its value is a unique integer label.
    c)split training and validation data.
    d)save your self.char as pickle (pkl) file that you may use later.
    d)map all characters of training and validation data to their integer label and save as 'npy' files respectively.

3)helper function to load preprocessed data

4)helper functions to create training and validation mini batches


"""


class TextLoader():
    def __init__(self, data_dir, batch_size, sequence_length, num_train=5, use_onehot=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_train = num_train # split training data, too much mem
        self.use_onehot = use_onehot
        self.preprocess_flag = 0

    def preprocess_data(self):
        """
        Read the shakespeare.txt file, assumed to be in data_dir
        """

        # read in the text
        with(open(self.data_dir + "/shakespeare.txt", 'rb')) as text:
            all_text = text.read().decode('utf8')
        
        # get chars
        all_text_list = list(all_text)

        # get unique characters
        self.char = set(all_text_list)

        # get number of them
        self.vocab_size = len(self.char)

        # create dictionary
        self.vocab = dict(zip(self.char, range(self.vocab_size)))
        
        cPickle.dump(self.char, open("s_char.pkl", 'wb'))
        cPickle.dump(self.vocab, open("s_vocab.pkl", 'wb'))

        # get training and validation data
        # we store as list of lists
        # each list contains sequence_length+1 char, where last char is
        # is to be predicted
        self.sequences=[]
        for i in range(self.sequence_length, len(all_text_list)):
            self.sequences.append(all_text_list[i - self.sequence_length:i+1])

        # map to integers via the dictionary
        self.sequence_map=[]
        for seq in self.sequences:
            self.sequence_map.append([self.vocab[x] for x in seq])

        # onehot
        split=len(self.sequence_map) * 95 // 100
        train_split = split // self.num_train
        # we split training into multiple files
        for i in range(self.num_train): 
            if self.use_onehot:
                train = [np.eye(len(self.char))[x] for x in self.sequence_map[i*train_split:(i+1)*train_split]]
                np.save('s_train_' + str(i) + '.npy', train)
            else:
                print('One hot encode is required')
        # save val
        val = [np.eye(len(self.char))[x] for x in self.sequence_map[split:]]
        np.save('s_val.npy', val)
        

    def load_train_data(self, num):
        """
        Loads the training data according to num
        Cannot load all into memory, esp with one-hot
        """
        self.train = np.load('s_train_' + str(num) + '.npy')

        return self.train
    
    def load_vocab(self):
        vocab = cPickle.load(open("s_vocab.pkl", "rb"))
        return vocab


    def validation_batch(self):
        # dont need to do this
        return 0
    
    def training_batch(self, batch_size):
        """
        Training batch generator
        """
        start = 0
        end = batch_size
        while end < len(self.train):
            yield self.train[start:end,:-1,:], self.train[start:end,-1,:]
            start += batch_size
            end += batch_size
        
        





