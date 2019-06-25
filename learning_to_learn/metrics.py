import multiprocessing as mp

import numpy as np
import tensorflow as tf

import learning_to_learn.tensors as tensors


def _mean_neuron_entropy_100(values):
    values.reshape((-1, values.shape[-1]))
    activations = tf.placeholder(tf.float32)
    entropy = tensors.mean_neuron_entropy_100_tf(activations)
    with tf.Session as sess:
        return sess.run(entropy, feed_dict={activations: values})


def mean_neuron_entropy_100(values):
    values = np.concatenate(values, 0)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=1) as pool:
        return pool.map(_mean_neuron_entropy_100, (values,))


def mean_mutual_information_100(values):
    pass
