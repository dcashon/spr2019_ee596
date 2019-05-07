import numpy as np
def mini_batch(features,labels,mini_batch_size, shuffle=True):
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