import matplotlib.pyplot as plt
import seaborn as sns
from HW1_mnist_function import calc_model_perf
import tensorflow as tf
import numpy as np
sns.set()


# D. Cashon
# 04-15-2019
# experiment with different model architecture and hyperparameters
# to improve performance on MNIST dataset
# modified interatively and plots generated

# setting up hyperparameters, etc to vary 
batch_sizes = [2**n for n in range(4, 8)]
possible_archs = [[256], [256]*2, [256]*3, [256]*5, [256]*8]
learning_rates = [1, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]
funcs = [tf.nn.sigmoid, tf.nn.relu, tf.nn.tanh]
n_count_vary = [[64, 64], [128,128], [256,256], [512, 512]]

# found "best" architecture, average some runs (range(10))

# what to vary
v_dict = {}
for params in range(10):
    arch, iterations, train, test = calc_model_perf(64, 0.01, [128, 128], 1000, tf.nn.sigmoid)
    v_dict[str(params)] = [arch, iterations, train, test]

# result of varying batch size
plt.figure(1, figsize=(15,10))
for key in list(v_dict.keys()):
    plt.plot(v_dict[key][1], v_dict[key][2], label='Train, run = ' + key)
    plt.plot(v_dict[key][1], v_dict[key][3], label='Test, run = ' + key, linestyle='dotted')
plt.xlabel('Iteration')
plt.ylabel('Sample Batch Accuracy (Test/Train)')
plt.title("Average Results for determined architecture" +  "\n" + str(arch))
plt.ylim(0,1)
plt.savefig('final.png', dpi=350, bbox_inches='tight')
plt.plot(np.linspace(0,1000,100), np.ones(100) * 0.96, linewidth='5', label='96 threshold')
plt.legend()
plt.show()