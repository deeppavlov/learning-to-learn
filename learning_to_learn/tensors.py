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


def get_probabilities_from_histograms(histograms, axis):
    with tf.name_scope('probabilities_from_histograms'):
        n = tf.reduce_sum(histograms, axis=axis, keepdims=True)
        probabilities = tf.cast(histograms, tf.float32) / tf.cast(n, tf.float32)
        return probabilities


def entropy_MLE_from_prob(probabilities, axis, keepdims=False):
    with tf.name_scope('entropy_MLE_from_prob'):
        log_prob = tf.log(probabilities) / tf.log(2.)
        log_prob = tf.where(
            tf.logical_or(tf.is_nan(log_prob), tf.is_inf(log_prob)),
            x=tf.zeros(tf.shape(log_prob)),
            y=log_prob
        )
        products = probabilities * log_prob
        return -tf.reduce_sum(products, axis=axis, keepdims=keepdims)


def entropy_MLE_from_hist(histograms, axis, keepdims=False):
    with tf.name_scope('entropy_MLE_from_hist'):
        probabilities = get_probabilities_from_histograms(histograms, axis)
        return entropy_MLE_from_prob(probabilities, axis, keepdims=keepdims)


def get_sample_size_and_support_from_hist(
        histograms,
        axis,
        keepdims=False,
        dtype=tf.float32
):
    with tf.name_scope('get_sample_size_and_support_from_hist'):
        n = tf.reduce_sum(histograms, axis=axis, keepdims=keepdims)
        m = tf.count_nonzero(histograms, axis=axis, keepdims=keepdims)
        n = tf.cast(n, dtype)
        m = tf.cast(m, dtype)
    return n, m


def entropy_MM_from_hist(histograms, axis, keepdims=False):
    with tf.name_scope('entropy_MM_from_hist'):
        n, m = get_sample_size_and_support_from_hist(
            histograms, axis, keepdims=True)
        entropy = entropy_MLE_from_hist(histograms, axis, keepdims=True) + (m - 1.) / (2. * n)
        if keepdims:
            return entropy
        return tf.squeeze(entropy, axis=axis)


def entropy_MM_from_prob(probabilities, n, m, axis, keepdims=False):
    with tf.name_scope('entropy_MM_from_prob'):
        n = tf.cast(n, tf.float32)
        m = tf.cast(m, tf.float32)
        entropy = entropy_MLE_from_prob(probabilities, axis, keepdims=True) + (m - 1.) / (2. * n)
        if keepdims:
            return entropy
        return tf.squeeze(entropy, axis=axis)


def sort_2_tf_values(value_1, value_2):
    with tf.name_scope('sort_2_tf_values'):
        first_value, second_value = tf.cond(
            tf.greater(value_1, value_2),
            true_fn=lambda: [value_2, value_1],
            false_fn=lambda: [value_1, value_2],
        )
        return first_value, second_value


def permute_two_axes(tensor, axis_1, axis_2):
    with tf.name_scope('permute_two_axes'):
        num_dims = tf.shape(tf.shape(tensor))[0]
        axis_1 %= num_dims
        axis_2 %= num_dims
        first_axis, second_axis = sort_2_tf_values(axis_1, axis_2)
        range_ = tf.range(num_dims)
        false_value = tf.concat(
            [
                range_[:first_axis],
                tf.reshape(second_axis, [1]),
                range_[first_axis + 1:second_axis],
                tf.reshape(first_axis, [1]),
                range_[second_axis + 1:]
            ],
            0
        )
        permutation = tf.cond(
            tf.equal(axis_1, axis_2),
            true_fn=lambda: range_,
            false_fn=lambda: false_value,
        )
        return tf.transpose(tensor, perm=permutation)


class PermuteTwoAxes:
    def __init__(self, tensor, axis_1, axis_2=-1):
        self.tensor = tensor
        self._axis_1 = axis_1
        self._axis_2 = axis_2

    def __enter__(self):
        self.tensor = permute_two_axes(self.tensor, self._axis_1, self._axis_2)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.tensor = permute_two_axes(self.tensor, self._axis_1, self._axis_2)


def shift_axis(tensor, axis, position):
    with tf.name_scope('shift_axis'):
        num_dims = tf.shape(tf.shape(tensor))[0]
        range_ = tf.range(num_dims)
        axis %= num_dims
        position %= num_dims
        first_axis, second_axis = sort_2_tf_values(axis, position)
        moved_dims = tf.reshape(axis, [1])
        fill_dims = tf.zeros([0], dtype=tf.int32)
        one = tf.constant(1)
        zero = tf.constant(0)
        first_dims, second_dims, before_2nd, after_1st, after_2nd = tf.cond(
            tf.greater(axis, position),
            true_fn=lambda: [moved_dims, fill_dims, zero, zero, one],
            false_fn=lambda: [fill_dims, moved_dims, one, one, one],
        )
        false_value = tf.concat(
            [
                range_[:first_axis],
                first_dims,
                range_[first_axis + after_1st:second_axis + before_2nd],
                second_dims,
                range_[second_axis + after_2nd:]
            ],
            0
        )
        permutation = tf.cond(
            tf.equal(axis, position),
            true_fn=lambda: range_,
            false_fn=lambda: false_value,
        )
        return tf.transpose(tensor, perm=permutation)


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


def get_slice_specs(shape, axis, idx, sample_size):
    with tf.name_scope('get_slice_specs'):
        shape = tf.identity(shape)
        num_dims = tf.shape(shape)[0]
        axis %= num_dims
        n = shape[axis]
        zeros = tf.zeros(tf.reshape(num_dims, [1]), dtype=tf.int32)
        start = tf.concat(
            [zeros[:axis], tf.reshape(idx, [1]), zeros[axis + 1:]],
            0
        )
        size = tf.concat(
            [
                shape[:axis],
                tf.reshape(tf.minimum(sample_size, n - 1 - idx), [1]),
                shape[axis + 1:]
            ],
            0
        )
        return start, size


def hist_1d_loop(values, num_bins, range_, axis, max_sample_size_per_iteration):
    with tf.name_scope('hist_1d_loop'):
        shape = tf.shape(values)
        output_shape = get_output_shape_for_hist_1d(values, axis, num_bins)
        n = shape[axis]
        i0 = tf.constant(0)
        hist = tf.zeros(output_shape, dtype=tf.int32)

        def body(idx, hist):
            start, size = get_slice_specs(tf.shape(values), axis, idx, max_sample_size_per_iteration)
            tensor = tf.slice(values, start, size)
            hist += hist_1d(tensor, num_bins, range_, axis)
            return idx + max_sample_size_per_iteration, hist

        _, hist = tf.while_loop(
            lambda x, y: tf.less(x, n),
            body,
            [i0, hist],
            shape_invariants=[tf.TensorShape([]), tf.TensorShape(None)],
            parallel_iterations=1,
        )
        return tf.reshape(hist, output_shape)


def hist_1d(values, num_bins, range_, axis):
    with tf.name_scope('hist_1d'):
        discrete = tf.histogram_fixed_width_bins(values, range_, num_bins)
        return hist_from_nonnegative_ints(discrete, axis, num_bins)


def compute_probabilities(activations, num_bins, range_, axis):
    with tf.name_scope('compute_probabilities'):
        n = tf.shape(activations)[axis]
        histograms = hist_1d(activations, num_bins, range_, axis)
        probabilities = tf.cast(histograms, tf.float32) / tf.cast(n, tf.float32)
        return probabilities


def mean_neuron_entropy_with_digitalization(activations, axis, num_bins, range_):
    with tf.name_scope('mean_neuron_entropy_with_digitalization'):
        histograms = hist_1d(activations, num_bins, range_, axis)
        return tf.reduce_mean(entropy_MM_from_hist(histograms, axis))


def neuron_entropy_with_digitalization(activations, axis, num_bins, range_):
    with tf.name_scope('neuron_entropy_with_digitalization'):
        histograms = hist_1d(activations, num_bins, range_, axis)
        return entropy_MM_from_hist(histograms, axis)


def self_cross_sum(tensor, axis):
    with tf.name_scope('self_cross_sum'):
        num_dims = tf.shape(tf.shape(tensor))[0]
        axis %= num_dims
        t1 = tf.expand_dims(tensor, axis=axis)
        t2 = tf.expand_dims(tensor, axis=axis + 1)
        return t1 + t2


def self_cross_sum_with_factors(tensor, axis, f1, f2):
    with tf.name_scope('self_cross_sum_with_factors'):
        num_dims = tf.shape(tf.shape(tensor))[0]
        axis %= num_dims
        t1 = tf.expand_dims(tensor, axis=axis)
        t2 = tf.expand_dims(tensor, axis=axis + 1)
        return f1 * t1 + f2 * t2


def get_output_shape_for_hist_1d(tensor, axis, num_bins):
    with tf.name_scope('get_output_shape_for_hist_1d'):
        shape = tf.shape(tensor)
        axis %= tf.shape(shape)[0]
        return tf.concat(
            [shape[:axis], tf.reshape(num_bins, [1]), shape[axis + 1:]],
            0
        )


def memory_efficient_bincount(values, num_bins):
    with tf.name_scope('memory_efficient_bincount'):
        data = tf.ones(tf.shape(values), dtype=tf.int32)
        return tf.unsorted_segment_sum(data, values, num_bins)


def hist_from_nonnegative_ints(tensor, axis, num_bins):
    output_shape = get_output_shape_for_hist_1d(tensor, axis, num_bins)
    with tf.name_scope('hist_from_nonnegative_ints'):
        with PermuteTwoAxes(tensor, axis) as permute_ctx:
            with TensorToMatrix(permute_ctx.tensor) as matrix_ctx:
                shape = tf.shape(matrix_ctx.tensor)
                n = shape[0]
                shifts = tf.reshape(num_bins * tf.range(n), [-1, 1])
                prepared_values = shifts + matrix_ctx.tensor
                nb = num_bins * n
                bc = memory_efficient_bincount(prepared_values, nb)
                backward_shape = tf.concat([shape[:1], [-1]], 0)
                matrix_ctx.tensor = tf.reshape(bc, backward_shape)
            permute_ctx.tensor = matrix_ctx.tensor
        return tf.reshape(permute_ctx.tensor, output_shape)


def cross_hist_from_tensor(tensor, num_bins, range_):
    with tf.name_scope('cross_hist_from_tensor'):
        bins = tf.histogram_fixed_width_bins(
            tensor,
            range_,
            nbins=num_bins,
        )
        bins_2d = self_cross_sum_with_factors(bins, -2, 1, num_bins)
        returned_value = hist_from_nonnegative_ints(bins_2d, -1, num_bins ** 2)
        return returned_value


def add_cross_hist_1_slice_independent(
        idx,
        histograms,
        activations,
        num_bins,
        range_,
        max_sample_size_per_iteration
):
    with tf.name_scope('add_cross_hist_1_slice_independent'):
        msspi = max_sample_size_per_iteration
        tensor = activations[..., idx:idx + msspi]
        histograms += cross_hist_from_tensor(tensor, num_bins, range_)
        return idx + msspi, histograms


def get_init_shape_of_cross_histograms(
        activations,
        num_bins
):
    shape = tf.shape(activations)
    cross_dim = tf.reshape(shape[-2], [1])
    return tf.concat(
        [
            shape[:-2],
            cross_dim,
            cross_dim,
            tf.reshape(num_bins ** 2, [1])
        ],
        0
    )


def sum_self_cross_histograms(
        activations,
        num_bins,
        range_,
        max_sample_size_per_iteration,
):
    with tf.name_scope('sum_self_cross_histograms'):
        def add_cross_hist_1_slice(idx, hists):
            return add_cross_hist_1_slice_independent(
                idx,
                hists,
                activations,
                num_bins,
                range_,
                max_sample_size_per_iteration
            )

        i0 = tf.constant(0)
        n = tf.shape(activations)[-1]
        histograms = tf.zeros(
            get_init_shape_of_cross_histograms(
                activations, num_bins),
            dtype=tf.int32
        )
        _, histograms = tf.while_loop(
            lambda x, y: x < n,
            add_cross_hist_1_slice,
            [i0, histograms],
            shape_invariants=[
                tf.TensorShape([]),
                histograms.get_shape()
            ],
            back_prop=False,
            parallel_iterations=1,
        )
        return histograms


def get_cross_histograms_permutation(
        num_dims,
        value_axis,
        cross_axis,
):
    value_axis %= num_dims
    cross_axis %= num_dims
    dims = tf.range(num_dims + 1)
    first_dims, second_dims, first_axis, second_axis = tf.cond(
        tf.greater(value_axis, cross_axis),
        true_fn=lambda: [dims[-3:-1], dims[-1:], cross_axis, value_axis],
        false_fn=lambda: [dims[-1:], dims[-3:-1], value_axis, cross_axis]
    )
    return tf.concat(
        [
            dims[:first_axis],
            first_dims,
            dims[first_axis:second_axis - 1],
            second_dims,
            dims[second_axis - 1:-3]
        ],
        0
    )


def get_self_cross_histograms(
        activations,
        value_axis,
        cross_axis,
        num_bins,
        range_,
        max_sample_size_per_iteration=2*10**4,
):
    with tf.name_scope('get_self_cross_histograms'):
        if max_sample_size_per_iteration is None:
            max_sample_size_per_iteration = int(2.5e8) // num_bins
        num_dims = tf.shape(tf.shape(activations))[0]
        value_axis %= num_dims
        cross_axis %= num_dims
        output_permutation = get_cross_histograms_permutation(
            num_dims,
            value_axis,
            cross_axis,
        )
        activations = shift_axis(activations, cross_axis, -1)
        new_value_axis = value_axis - tf.cast(value_axis > cross_axis, tf.int32)
        activations = shift_axis(activations, new_value_axis, -1)
        histograms = sum_self_cross_histograms(
            activations,
            num_bins,
            range_,
            max_sample_size_per_iteration,
        )
        return tf.transpose(histograms, perm=output_permutation)


def get_min_nonzero(tensor):
    with tf.name_scope('get_min_nonzero'):
        tensor = tf.reshape(tensor, [-1])
        unique, _ = tf.unique(tensor)
        top2, _ = tf.math.top_k(-unique, 2)
        top1, idx = tf.math.top_k(top2, 1)
        return tf.where(tf.equal(top1[0], 0), x=-top2[idx[0] - 1], y=-top1[0])


def squeeze_tf(tensor, axis):
    with tf.name_scope('squeeze_tf'):
        shape = tf.shape(tensor)
        new_shape = tf.concat([shape[:axis], shape[axis + 1:]], 0)
        return tf.reshape(tensor, new_shape)


def mutual_information_from_hists(
        cross_hist,
        hist,
        value_axis_2d,
        value_axis,
        cross_axis,
        keepdims=False,
):
    with tf.name_scope('mutual_information_from_hists'):
        entropy = entropy_MM_from_hist(hist, value_axis, keepdims=True)
        entropy_sum = self_cross_sum(entropy, cross_axis)
        joint_entropy = entropy_MM_from_hist(cross_hist, value_axis_2d, keepdims=True)
        mutual_info = entropy_sum - joint_entropy
        if keepdims:
            return mutual_info
        return squeeze_tf(mutual_info, value_axis_2d)


def mutual_information_and_min_nonzero_count(
        activations,
        value_axis,
        cross_axis,
        num_bins,
        range_,
        keepdims=False,
        max_sample_size_per_iteration=2*10**4,
):
    with tf.name_scope('mutual_information_and_min_nonzero_count'):
        num_dims = tf.shape(tf.shape(activations))[0]
        value_axis %= num_dims
        cross_axis %= num_dims
        histograms = hist_1d_loop(activations, num_bins, range_, value_axis, 10 ** 6)
        cross_histograms = get_self_cross_histograms(
            activations,
            value_axis,
            cross_axis,
            num_bins,
            range_,
            max_sample_size_per_iteration=max_sample_size_per_iteration,
        )
        value_axis_2d = tf.cast(value_axis > cross_axis, tf.int32) + value_axis
        mutual_info = mutual_information_from_hists(
            cross_histograms,
            histograms,
            value_axis_2d,
            value_axis,
            cross_axis,
            keepdims=True,
        )
        min_nonzero = get_min_nonzero(cross_histograms)
        if keepdims:
            return mutual_info, min_nonzero
        else:
            return squeeze_tf(mutual_info, value_axis), min_nonzero


def mean_mutual_information_and_min_nonzero_count(
        activations,
        value_axis,
        cross_axis,
        num_bins,
        range_,
        keepdims=False,
        max_sample_size_per_iteration=2*10**4,
):
    with tf.name_scope('mean_mutual_information_and_min_nonzero_count'):
        num_dims = tf.shape(tf.shape(activations))[0]
        value_axis %= num_dims
        cross_axis %= num_dims
        mutual_info, min_nonzero = mutual_information_and_min_nonzero_count(
            activations,
            value_axis,
            cross_axis,
            num_bins,
            range_,
            keepdims=True,
            max_sample_size_per_iteration=max_sample_size_per_iteration,
        )
        value_axis = value_axis + tf.cast(tf.greater(value_axis, cross_axis), tf.int32)
        if not keepdims:
            mutual_info = squeeze_tf(mutual_info, value_axis)
        mutual_info_reshaped = shift_axis(mutual_info, cross_axis + 1, -1)
        mutual_info_reshaped = shift_axis(mutual_info_reshaped, cross_axis, -1)
        diag = tf.linalg.diag_part(mutual_info_reshaped)
        N = tf.reduce_prod(tf.shape(mutual_info_reshaped))
        n = tf.reduce_prod(tf.shape(diag))
        s = tf.reduce_sum(mutual_info_reshaped) - tf.reduce_sum(diag)
        return mutual_info, s / tf.cast(N - n, tf.float32), min_nonzero

