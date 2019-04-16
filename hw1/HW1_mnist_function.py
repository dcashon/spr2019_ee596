import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#get mnist data, with one_hot encoding
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# D. Cashon
# 04-14-2019
# function to vary parameters of MNIST model

def calc_model_perf(b_size, lr, num_n, num_iter, act_func):
    """
    Function to allow for easily varying the model parameters.
    Assumption is that only one parameter is a list of values to try...

    Parameters:
        b_size:     batch size
        lr:         learning rate
        num_n:      num of neurons in each layer, list
        num_iter:   num_iter to train, mostly for speed
        act_func:   activation function
    
    Outputs:
        test_acc:   model testing accuracy on random batch
        train_acc:  model training accuracy on batch

    """
    # model architecture:
    arch = {'batch_size': b_size, 'learning_rate': lr, 'layers': num_n, 'num_iter': num_iter, 
            'act_func': act_func}

    # get the data
    train_imgs = mnist.train.images
    train_labels = mnist.train.labels
    test_imgs = mnist.test.images
    test_labels = mnist.test.labels

    # 
    num_layers = len(num_n)

    # construct the graph
    tf.reset_default_graph()
    x_in = tf.placeholder(dtype=tf.float32, shape=[None, 784])

    # define true labels
    y_hat = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    weights, biases = {}, {}
    for i in range(num_layers):
        wkey = 'W' + str(i)
        bkey = 'B' + str(i)
        if i == 0:
            # first layer, use input dim
            weights[wkey] = tf.get_variable(wkey, shape=[784, num_n[i]])
            biases[bkey] = tf.get_variable(bkey, shape=[1,1])
        else:
            # other layers, use dim and dim-1
            weights[wkey] = tf.get_variable(wkey, shape=[num_n[i-1], num_n[i]])
            biases[bkey] = tf.get_variable(bkey, shape=[1,1])

    # add output layer weight and bias
    wkey_out = 'W' + str(num_layers)
    bkey_out = 'B' + str(num_layers)
    weights[wkey_out] = tf.get_variable(wkey_out, shape=[num_n[num_layers-1], 10])
    biases[bkey_out] = tf.get_variable(bkey_out, shape=[1,1])
    
    # for now sigmoid activation, but will try others too
    layers_out = []
    for i, wb in enumerate(list(zip(weights, biases))):
        if i == 0:
            # first layer, needs x_in
            layers_out.append(act_func(tf.add(tf.matmul(x_in, weights[wb[0]]), biases[wb[1]])))
        else:
            layers_out.append(act_func(tf.add(tf.matmul(layers_out[i-1], weights[wb[0]]), biases[wb[1]])))
    logits = layers_out[-1]

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_hat))
    #define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(cost)

    #compare the predicted labels with true labels
    correct_pred = tf.equal(tf.argmax(logits,1),tf.argmax(y_hat,1))

    #compute the accuracy by taking average
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='accuracy')

    # compute
    init = tf.global_variables_initializer()
    iterations, train_acc, test_acc = [], [], []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_iter):
            # assuming we dont get to the end before num_iterationss finishes
            batch_start = i * b_size
            batch_end = i * b_size + b_size
            # train
            sess.run(train_op, feed_dict={x_in: train_imgs[batch_start:batch_end, :], y_hat: train_labels[batch_start:batch_end, :]})
            if i % 100 == 0:
                # print accuracy
                acc = sess.run(accuracy, feed_dict={x_in: train_imgs[batch_start:batch_end, :], y_hat: train_labels[batch_start:batch_end, :]})
                cst = sess.run(cost, feed_dict={x_in: train_imgs[batch_start:batch_end, :], y_hat: train_labels[batch_start:batch_end, :]})
                test_a = sess.run(accuracy, feed_dict={x_in: test_imgs[b_size:2*b_size, :], y_hat: test_labels[b_size:2*b_size, :]})
                print('Sample Test Accuracy \t ' + str(test_a))
                print('Accuracy \t ' + str(acc))
                print('Cost \t' + str(cst))
                iterations.append(i)
                train_acc.append(acc)
                test_acc.append(test_a)

    return arch, iterations, train_acc, test_acc

