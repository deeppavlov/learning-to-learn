import tensorflow as tf
from meta import Meta


class ResNet4Lstm(Meta):

    def __init__(self,
                 optimizee,
                 num_exercises=10,
                 num_lstm_nodes=256,
                 num_optimizer_unrollings=10,
                 perm_period=None,
                 num_gpus=1,
                 regime='train'):
        self._optimizee = optimizee
        self._num_exercises = num_exercises
        self._num_lstm_nodes = num_lstm_nodes
        self._num_optimizer_unrollings = num_optimizer_unrollings
        self._perm_period = perm_period
        self._num_gpus = num_gpus
        self._regime = regime

        self._hooks = dict(
            grad_eval_inputs=None,
            grad_eval_labels=None,
            optimizer_learn_inputs=None,
            optimizer_learn_labels=None,
            opt_inference_inputs=None,
            opt_inference_labels=None
        )

        if regime == 'train':
            ex_per_gpu = self._num_exercises // self._num_gpus
            remaining = self._num_exercises - self._num_gpus * ex_per_gpu
            self._exercise_gpu_map = [n // ex_per_gpu for n in range((self._num_gpus - 1) * ex_per_gpu)] + \
                                     [self._num_gpus - 1] * (ex_per_gpu + remaining)

            tmp = self._make_inputs_and_labels_placeholders(
                self._optimizee, self._num_optimizer_unrollings, self._num_exercises,
                self._exercise_gpu_map, regime='train')
            self._grad_eval_inputs, self._grad_eval_labels,\
                self._optimizer_learn_inputs, self._optimizer_learn_labels = tmp
        else:
            self._exercise_gpu_map = None
            self._grad_eval_inputs, self._grad_eval_labels, \
                self._optimizer_learn_inputs, self._optimizer_learn_labels = None, None, None, None

        self._opt_inference_inputs, self._opt_inference_labels = self._make_inputs_and_labels_placeholders(
            self._optimizee, self._num_optimizer_unrollings, None, None, regime='inference')

        self._hooks['grad_eval_inputs'] = self._grad_eval_inputs
        self._hooks['grad_eval_labels'] = self._grad_eval_labels
        self._hooks['optimizer_learn_inputs'] = self._optimizer_learn_inputs
        self._hooks['optimizer_learn_labels'] = self._optimizer_learn_labels
        self._hooks['opt_inference_inputs'] = self._opt_inference_inputs
        self._hooks['opt_inference_labels'] = self._opt_inference_labels