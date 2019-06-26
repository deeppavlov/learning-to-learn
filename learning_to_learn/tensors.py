import numpy as np
import tensorflow as tf

from learning_to_learn.useful_functions import InvalidArgumentError


def choose_biggest(a, b, name_scope):
    with tf.name_scope(name_scope):
        mask = tf.to_float(a > b)
        return mask * a + (1. - mask) * b


def metrics_reduce_mean(metrics, keep_first_dim, metrics_name):
    reduce_axes = tf.range(1, tf.shape(tf.shape(metrics))[0], delta=1, dtype=tf.int32)
    accuracy = tf.reduce_mean(metrics, axis=reduce_axes, name='%s_keep_first_dim' % metrics_name)
    if not keep_first_dim:
        metrics = tf.reduce_mean(accuracy, name=metrics_name)
    return metrics


def perplexity_tensor(probabilities=None, keep_first_dim=False):
    with tf.name_scope('computing_perplexity'):
        ln2 = np.log(2, dtype=np.float32)
        # shape = probabilities.get_shape().as_list()
        probabilities = choose_biggest(probabilities, 1e-10, 'to_small_values_in_probs_are_filtered')
        # probabilities = tf.where(probabilities > 1e-10,
        #                          probabilities,
        #                          np.full(tuple(shape), 1e-10),
        #                          name='to_small_values_in_probs_are_filtered')
        log_probabilities = tf.divide(tf.log(probabilities), ln2, name='log2_probs')
        entropy = tf.reduce_sum(- probabilities * log_probabilities, axis=-1, name='entropy_not_mean')
        perplexity = tf.exp(ln2 * entropy, name='perplexity_not_aver')
        return metrics_reduce_mean(perplexity, keep_first_dim, 'perplexity')


def loss_tensor(predictions=None, labels=None, keep_first_dim=False):
    with tf.name_scope('computing_loss'):
        # shape = predictions.get_shape().as_list()
        predictions = choose_biggest(predictions, 1e-10, 'to_small_values_in_probs_are_filtered')
        # predictions = tf.where(predictions > 1e-12,
        #                        predictions,
        #                        tf.constant(1e-12),
        #                        name='to_small_values_in_probs_are_filtered')
        log_predictions = tf.log(predictions, name='log_pred')

        loss_on_characters = tf.reduce_sum(-labels * log_predictions, axis=-1, name='loss_not_mean')
        return metrics_reduce_mean(loss_on_characters, keep_first_dim, 'loss_on_characters')


def bpc_tensor(loss=None):
    with tf.name_scope('computing_bpc'):
        return loss / np.log(2)


def accuracy_tensor(predictions=None, labels=None, keep_first_dim=False):
    with tf.name_scope('computing_accuracy'):
        predictions = tf.argmax(predictions, axis=-1, name='predictions')
        # print("(learning_to_learn.useful_functions.accuracy_tensor)labels:", labels)
        labels = tf.argmax(labels, axis=-1, name='labels')

        # predictions = tf.Print(
        #     predictions,
        #     [predictions],
        #     message='predictions_in_accuracy:', summarize=1200)
        # labels = tf.Print(labels, [labels], message='labels_in_accuracy:', summarize=1200)

        accuracy = tf.to_float(tf.equal(predictions, labels), name='accuracy_not_averaged')
        return metrics_reduce_mean(accuracy, keep_first_dim, 'accuracy')


def identity_tensor(**kwargs):
    if len(kwargs) > 1:
        raise InvalidArgumentError('kwargs should not contain 1 entry', kwargs, 'kwargs', 'len(kwargs)=1')
    for value in kwargs.values():
        return value


def compute_metrics(metrics, predictions=None, labels=None, loss=None, keep_first_dim=False):
    # print("(tensors.compute_metrics)predictions.shape:", predictions.get_shape().as_list())
    # print("(tensors.compute_metrics)labels.shape:", labels.get_shape().as_list())
    # print("(tensors.compute_metrics)loss.shape:", loss.get_shape().as_list())
    with tf.name_scope('compute_metrics'):
        res = dict()
        if 'loss' in metrics:
            l = loss_tensor(predictions=predictions, labels=labels, keep_first_dim=keep_first_dim)
            res['loss'] = l
        else:
            l = None
        if 'bpc' in metrics:
            if loss is not None:
                bpc = bpc_tensor(loss=loss)
                res['bpc'] = bpc
            elif l is not None:
                bpc = bpc_tensor(loss=l)
                res['bpc'] = bpc
            elif predictions is not None and labels is not None:
                bpc = bpc_tensor(loss=loss_tensor(predictions=predictions, labels=labels, keep_first_dim=keep_first_dim))
                res['bpc'] = bpc
            else:
                print('loss:', loss)
                print('metrics:', metrics)
                print('predictions:', predictions)
                print('labels:', labels)
                raise InvalidArgumentError(
                    'Could not build bpc graph. Not enough arguments were provided.',
                    [metrics, predictions, labels, loss],
                    ['metrics', 'predictions', 'labels', 'loss'],
                    'At least loss or predictions and labels has to be not None'
                )
        if 'accuracy' in metrics:
            accuracy = accuracy_tensor(predictions=predictions, labels=labels, keep_first_dim=keep_first_dim)
            res['accuracy'] = accuracy
        if 'perplexity' in metrics:
            perplexity = perplexity_tensor(probabilities=predictions, keep_first_dim=keep_first_dim)
            res['perplexity'] = perplexity
        return res


def compute_metrics_raw_lbls(metrics, predictions=None, labels=None, loss=None, keep_first_dim=False):
    voc_size = tf.shape(predictions)[-1]
    labels = tf.one_hot(labels, voc_size, dtype=tf.float32)
    # print("(learning_to_learn.useful_functions.compute_metrics_raw_lbls)labels:", labels)
    labels = tf.reshape(labels, tf.shape(predictions))
    # print("(learning_to_learn.useful_functions.compute_metrics_raw_lbls)labels:", labels)
    return compute_metrics(metrics, predictions=predictions, labels=labels, loss=loss, keep_first_dim=keep_first_dim)


def log_and_sign(inp, p):
    edge = np.exp(-p)
    mask1 = tf.to_float(tf.abs(inp) > edge)
    mask = tf.expand_dims(mask1, axis=-1)
    prep_for_log = mask1 * inp + (1. - mask1)
    greater_first = tf.log(tf.abs(prep_for_log)) / p
    greater_second = tf.sign(inp)
    less_first = tf.fill(tf.shape(inp), -1.)
    less_second = np.exp(p) * inp
    greater = tf.stack([greater_first, greater_second], axis=-1)
    less = tf.stack([less_first, less_second], axis=-1)
    res = mask * greater + (1. - mask) * less
    return res


def entropy_MM(probabilities, n, m):
    with tf.name_scope('entropy_MM'):
        log_prob = tf.log(probabilities) / np.log(2)
        products = probabilities * log_prob
        return -tf.reduce_sum(products, -1) + (m - 1) / (2 * n)


class AxisToTheBack:
    def __init__(self, tensor, axis):
        self.tensor = tensor
        self.axis = axis
        self._num_dims = None
        self._permutation = None

    def __enter__(self):
        self._num_dims = tf.shape(tf.shape(self.tensor))[0]
        range_ = tf.range(self._num_dims)
        false_value = tf.concat(
            [
                range_[:self.axis],
                tf.reshape(self._num_dims - 1, [1]),
                range_[self.axis + 1:-1],
                tf.reshape(self.axis, [1]),
            ],
            0
        )
        self._permutation = tf.cond(
            tf.equal(self._num_dims - 1, self.axis),
            true_fn=lambda: range_,
            false_fn=lambda: false_value,
        )
        self.tensor = tf.transpose(self.tensor, perm=self._permutation)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.tensor = tf.transpose(self.tensor, perm=self._permutation)


class TensorToMatrix:
    def __init__(self, tensor):
        self.tensor = tensor
        self._old_shape = None

    def __enter__(self):
        self._old_shape = tf.shape(self.tensor)
        self.tensor = tf.reshape(self.tensor, tf.stack([-1, self._old_shape[-1]]))
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.tensor = tf.reshape(self.tensor, tf.concat([self._old_shape[:-1], [-1]], 0))


def hist_1d(values, num_bins, range_, axis):
    with tf.name_scope('hist_1d'):
        def hist_1_vec(histograms, vec_idx, tensor):
            with tf.name_scope('hist_1_vec'):
                hist = tf.histogram_fixed_width(tensor[vec_idx], range_, nbins=num_bins)
                hist = hist[tf.newaxis, :]
                histograms = tf.concat([histograms, hist], 0)
                vec_idx += 1
                return histograms, vec_idx, tensor

        histograms = tf.zeros(tf.stack([0, num_bins]))
        vec_idx = tf.constant(0)
        with AxisToTheBack(values, axis) as axis_ctx:
            with TensorToMatrix(axis_ctx.tensor) as shape_ctx:
                tensor = shape_ctx.tensor
                histograms, _, _ = tf.while_loop(
                    lambda x, y, z: y < tf.shape(z)[-1],
                    hist_1_vec,
                    [histograms, vec_idx, tensor],
                    shape_invarivants=(
                        tf.TensorShape([None, None]),
                        [],
                        tensor.get_shape(),
                    ),
                    back_prop=False,
                )
        return histograms


def compute_probabilities(activations, num_bins, range_, axis):
    with tf.name_scope('compute_probabilities'):
        n = tf.shape(activations)[axis]
        histograms = hist_1d(activations, num_bins, range_, axis)
        probabilities = histograms / tf.cast(n, tf.float32)
        return probabilities


def mean_neuron_entropy_100_tf(activations):
    with tf.name_scope('mean_neuron_entropy_100_tf'):
        n = tf.shape(activations)[-1]
        num_bins = 100
        range_ = [-1., 1.]
        probabilities = compute_probabilities(activations, num_bins, range_, -1)
        return entropy_MM(probabilities, n, num_bins)

