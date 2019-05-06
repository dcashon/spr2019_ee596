# D. Cashon
# 2019 05 05
# functions for data manipulation of the dogs and cats dataset
import pickle
import numpy as np

def load_preprocessed_training_batch(batch_id,mini_batch_size):
    """
    Args:
        batch_id: the specific training batch you want to load
        mini_batch_size: the number of examples you want to process for one update
    Return:
        mini_batch(features,labels, mini_batch_size)
    """
    # this assumes that you are in the working dir with the 
    # preprocessed files, as path is hardcoded
    file_name = 'dog_cat_training_' + str(batch_id) + '.pkl'
    with open(file_name, 'rb') as foo:
        d1 = pickle.load(foo)
    features = d1['data']
    labels = d1['labels']

    #shuffle
    shuffle_idx = np.arange(0,len(features))
    np.random.shuffle(shuffle_idx) #inplace
    #print(shuffle_idx)
    return mini_batch(features[shuffle_idx],labels[shuffle_idx],mini_batch_size)

def mini_batch(features,labels,mini_batch_size):
    """
    Args:
        features: features for one batch
        labels: labels for one batch
        mini_batch_size: the mini-batch size you want to use.
    Hint: Use "yield" to generate mini-batch features and labels
    """
    counter = 0
    start_idx = 0
    end_idx = mini_batch_size
    while end_idx < len(labels):
        yield (features[start_idx:end_idx], labels[start_idx:end_idx])
        counter += 1
        start_idx = counter * mini_batch_size
        end_idx = counter * mini_batch_size + mini_batch_size


