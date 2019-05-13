## EE596: Practical Introduction to Neural Nets Homework 2

## Notes
All training was done using either a local 1050 Ti or Tesla P100 gpu on mox.hyak nodes.

### Problem 1:
Single fully connected hidden layer is implemented in a class inside:

NumpyNet.py

Both batch gradient descent and ADAM are implemented as options in the train method. See here for more info on ADAM:

<http://proceedings.mlr.press/v80/balles18a/balles18a.pdf>

Plots of performance using batch sizes [16, 64, 256, 1024] on MNIST are found in:

HW2_NumpyNet_BatchGD_ADAM_Tflow.ipynb

### Problem 2:

See:

load_cifar.py

For functions that process the CIFAR10 dataset. The neural network used is found in:

HW2_4b.ipynb

### Problem 3:

LeNet-5 on MNIST:

HW2_3a.ipynb

On CIFAR-10:

HW2_3b.ipynb

### Problem 4:

Functions reused from other problems and found in:






