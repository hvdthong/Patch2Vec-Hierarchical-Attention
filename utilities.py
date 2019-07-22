import math
import numpy as np


def mini_batches(X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_added = X_added_code
    shuffled_X_removed = X_removed_code
    shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_extended(X_ftr, X_msg, X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0):
    m = Y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X_ftr = X_ftr
    shuffled_X_msg = X_msg
    shuffled_X_added = X_added_code
    shuffled_X_removed = X_removed_code
    shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X_ftr = shuffled_X_ftr[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_msg = shuffled_X_msg[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_ftr, mini_batch_X_msg, mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
