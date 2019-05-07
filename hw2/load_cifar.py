import pickle
import numpy as np
from pathlib import Path

# D. Cashon
# 2019 04 24
# functions to load and preprocess CIFAR10 data
# Modifications:
#	pathlib useful for handling windows file paths more easily, and to ensure
#   cross compatibility between Linux.Mac


def load_training_batch(folder_path, batch_id):
    """
    Args:
        folder_path: the directory contains data files
        batch_id: training batch id (1,2,3,4,5)
    Return:
        features: numpy array that has shape (10000,3072)
        labels: a list that has length 10000
    """
    # create path
    cwd = Path(folder_path)
    batch_path = 'data_batch_' + str(batch_id)
    full_path = cwd / batch_path

    # load the data
    with open(full_path, 'rb') as foo:
        d1 = pickle.load(foo, encoding='bytes')

    # get the features
    features = d1[b'data']

    # get the labels and cast to np.array from list
    # this will be more convenient later
    labels = np.array(d1[b'labels'])

    return features, labels

# Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
    """
    Args:
        folder_path: the directory contains data files
    Return:
        features: numpy array that has shape (10000,3072)
        labels: a list that has length 10000
    """

    # path
    cwd = Path(folder_path)
    batch_path = 'test_batch'
    full_path = cwd / batch_path

    # load
    with open(full_path, 'rb') as foo:
        d1 = pickle.load(foo, encoding='bytes')

    # features
    features = d1[b'data']

    # labels (to np array)
    labels = np.array(d1[b'labels'])

    return features, labels

# Step 3: define a function that returns a list that contains label names (order is matter)
def load_label_names(folder_path):
    """
    Args:
        folder_path: full path to directory containing the data files
    Returns:
        labels: python list of class labels as strings
        airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    """
    # label names are in batches.meta
    meta_path = Path(folder_path) / 'batches.meta'
    with open(meta_path, 'rb') as foo:
        d1 = pickle.load(foo, encoding='utf8')
    
    labels = d1['label_names']

    return labels
    

# Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
    """
    Args:
        features: a numpy array with shape (10000, 3072)
    Return:
        features: a numpy array with shape (10000,32,32,3)
    """
    # order F allows reshape to place the channels in proper place
    # I assume this function is for viewing the images mainly

    return np.reshape(features, (10000, 32, 32, 3), order='F').copy()

# Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path, batch_id, data_id):
    """
    Args:
        folder_path: directory that contains data files
        batch_id: the specific number of batch you want to explore.
        data_id: the specific number of data example you want to visualize
    Return:
        None

    Descrption:
        1)You can print out the number of images for every class.
        2)Visualize the image
        3)Print out the minimum and maximum values of pixel
    """
    # may do later
    return 0

# Step 6: define a function that does min-max normalization on input
def normalize(x):
    """
    Args:
        x: features, a numpy array
    Return:
        x: normalized features, by column
    """
    x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

    return x

# Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
    """
    Args:
        x: a np array of labels in [0, 9]
    Return:
        a numpy array that has shape (len(x), # of classes)
    """
    # 10 classes is known
    one_hot = np.zeros((len(x), 10))
    for i, labels in enumerate(x):
        one_hot[i, labels] = 1
    
    return one_hot

# Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features, labels, filename):
    """
    Args:
        features: numpy array
        labels: a list of labels
        filename: the file you want to save the preprocessed data
    """
    # normalize
    x_data_norm = normalize(features)
    # one hot
    x_labels_onehot = one_hot_encoding(labels)
    # dictionary
    d1 = {'data': x_data_norm, 'labels_onehot': x_labels_onehot}
    #save
    with open(filename, 'wb') as foo:
        pickle.dump(d1, foo)

    return 0

# Step 9:define a function that preprocesss all training batch data and test data. 
# Use 10% of your total training data as your validation set
# In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
    """
    Args:
        folder_path: the directory contains your data files
    """
    # I assume the training data is already split randomly
    # so will just take 1000 from each training batch for the validation set
    # then training batch size will remain the same
    cwd = Path(folder_path)
    val_data, val_labels = [], []
    # save training data and split up validation data
    for i in range(1,6):
        data, labels = load_training_batch(cwd, i)
        preprocess_and_save(data[:9000], labels[:9000], 'train_processed_' + str(i) + '.pkl')
        val_data.append(data[9000:].copy())
        val_labels.append(labels[9000:].copy())
    # save validation data
    val_data_all = np.concatenate(val_data)
    val_labels_all = np.concatenate(val_labels)
    preprocess_and_save(val_data_all, val_labels_all, 'val_processed.pkl')
    # save test data
    test_data, test_labels = load_testing_batch(cwd)
    preprocess_and_save(test_data, test_labels, 'test_processed.pkl')

    return 0

# Step 10: define a function to yield mini_batch
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

# Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
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
    file_name = 'train_processed_' + str(batch_id) + '.pkl'
    with open(file_name, 'rb') as foo:
        d1 = pickle.load(foo)
    features = d1['data']
    labels = d1['labels_onehot']

    #shuffle
    shuffle_idx = np.arange(0,9000)
    np.random.shuffle(shuffle_idx) #inplace
    #print(shuffle_idx)
    return mini_batch(features[shuffle_idx, :],labels[shuffle_idx, :],mini_batch_size)

# Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch():
    file_name = 'val_processed.pkl'
    with open(file_name, 'rb') as foo:
        d1 = pickle.load(foo)
    features = d1['data']
    labels = d1['labels_onehot']

    return features, labels

# Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
    file_name = 'test_processed.pkl'
    with open(file_name, 'rb') as foo:
        d1 = pickle.load(foo)
    features = d1['data']
    labels = d1['labels_onehot']

    return mini_batch(features,labels,test_mini_batch_size)

