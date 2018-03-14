import tensorflow as tf
from meta import Meta


class ResNet4Lstm(Meta):

    def __init__(self,
                 pupil,
                 num_exercises=10,
                 num_lstm_nodes=256,
                 num_optimizer_unrollings=10,
                 perm_period=None,
                 num_gpus=1,
                 regime='train'):
        self._pupil = pupil
        self._num_exercises = num_exercises
        self._num_lstm_nodes = num_lstm_nodes
        self._num_optimizer_unrollings = num_optimizer_unrollings
        self._perm_period = perm_period
        self._num_gpus = num_gpus
        self._regime = regime

        self._hooks = dict(
            pupil_grad_eval_inputs=None,
            pupil_grad_eval_labels=None,
            optimizer_grad_inputs=None,
            optimizer_grad_labels=None,
            opt_inference_inputs=None,
            opt_inference_labels=None
        )

        if regime == 'train':
            ex_per_gpu = self._num_exercises // self._num_gpus
            remaining = self._num_exercises - self._num_gpus * ex_per_gpu
            self._exercise_gpu_map = [n // ex_per_gpu for n in range((self._num_gpus - 1) * ex_per_gpu)] + \
                                     [self._num_gpus - 1] * (ex_per_gpu + remaining)

            tmp = self._make_inputs_and_labels_placeholders(
                self._pupil, self._num_optimizer_unrollings, self._num_exercises,
                self._exercise_gpu_map)
            self._pupil_grad_eval_inputs, self._pupil_grad_eval_labels,\
                self._optimizer_grad_inputs, self._optimizer_grad_labels = tmp
            self._pupil_trainable_variables, self._pupil_grad_eval_pupil_storage, self._optimizer_grad_pupil_storage, \
            self._pupil_savers = self._create_pupil_variables_and_savers(
                self._pupil, self._num_exercises, self._exercise_gpu_map)
        else:
            self._exercise_gpu_map = None
            self._pupil_grad_eval_inputs, self._pupil_grad_eval_labels, \
                self._optimizer_grad_inputs, self._optimizer_grad_labels = None, None, None, None

        self._hooks['pupil_grad_eval_inputs'] = self._pupil_grad_eval_inputs
        self._hooks['pupil_grad_eval_labels'] = self._pupil_grad_eval_labels
        self._hooks['optimizer_grad_inputs'] = self._optimizer_grad_inputs
        self._hooks['optimizer_grad_labels'] = self._optimizer_grad_labels
        
        