import multiprocessing as mp

import numpy as np
import tensorflow as tf

import learning_to_learn.tensors as tensors


def _mean_neuron_entropy_100(values):
    values = values.reshape((-1, values.shape[-1]))
    activations = tf.placeholder(tf.float32)
    entropy = tensors.mean_neuron_entropy_with_digitalization(activations, 0, 100, [-1., 1.])
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        return sess.run(entropy, feed_dict={activations: values})


def _mean_mutual_information_100(values):
    values = values.reshape((-1, values.shape[-1]))
    activations = tf.placeholder(tf.float32)
    _, mutual_information, _ = \
        tensors.mean_mutual_information_and_min_nonzero_count(
            activations, 0, 1, 100, [-1., 1.])
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        return sess.run(mutual_information, feed_dict={activations: values})


def mean_neuron_entropy_100(values):
    values = np.concatenate(values, 0)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=1) as pool:
        return pool.map(_mean_neuron_entropy_100, (values,))[0]


def mean_mutual_information_100(values):
    values = np.concatenate(values, 0)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=1) as pool:
        return pool.map(_mean_mutual_information_100, (values,))[0]


def _entropy_metrics_100(values):
    values = values.reshape((-1, values.shape[-1]))
    activations = tf.placeholder(tf.float32)
    entropy = tensors.neuron_entropy_with_digitalization(
        activations,
        0,
        100,
        [-1., 1.],
    )
    mean_entropy = tf.reduce_mean(entropy)
    mutual_info, mean_mutual_info, support = tensors.mean_mutual_information_and_min_nonzero_count(
        activations,
        0,
        1,
        100,
        [-1., 1.],
    )
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        result = sess.run(
            [entropy, mean_entropy, mutual_info, mean_mutual_info, support],
            feed_dict={activations: values}
        )
        return result


def entropy_metrics_100(values):
    values = np.concatenate(values, 0)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=1) as pool:
        return pool.map(_entropy_metrics_100, (values,))[0]


def _min_value_of_nonzero_counts_100(values):
    values = values.reshape((-1, values.shape[-1]))
    activations = tf.placeholder(tf.float32)
    _, _, support = tensors.mean_mutual_information_and_min_nonzero_count(
        activations,
        0,
        1,
        100,
        [-1., 1.],
    )
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        support = sess.run(support, feed_dict={activations: values})
        return support


def min_value_of_nonzero_counts_100(values):
    values = np.concatenate(values, 0)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=1) as pool:
        return pool.map(_min_value_of_nonzero_counts_100, (values,))[0]
