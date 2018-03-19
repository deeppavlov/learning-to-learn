import tensorflow as tf
from meta import Meta


class ResNet4Lstm(Meta):

    def _create_optimizer_states(self):
        with tf.variable_scope('optimizer_states'):
            states = [
                tf.get_variable('h', tf.zeros([self._num_lstm_nodes, self._num_lstm_nodes])),
                tf.Variable('c', tf.zeros([self._num_lstm_nodes, self._num_lstm_nodes]))
            ]
            return states

    @staticmethod
    def _reset_optimizer_states():
        with tf.variable_scope('optimizer_states', resue=True):
            h = tf.get_variable('h')
            c = tf.get_variable('c')
            h_shape = h.get_shape.as_list()
            c_shape = c.get_shape().as_list()
            reset_ops = [
                tf.assign(h, tf.zeros(h_shape)),
                tf.assign(c, c_shape)
            ]
            return tf.group(*reset_ops)

    def _optimizer_core(self, optimizer_ins, states):


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
        if self._num_gpus == 1:
            self._base_device = '/gpu:0'
        else:
            self._base_device = '/cpu:0'
        self._regime = regime

        self._hooks = dict(
            pupil_grad_eval_inputs=None,
            pupil_grad_eval_labels=None,
            optimizer_grad_inputs=None,
            optimizer_grad_labels=None,
            pupil_savers=None,
            optimizer_train_op=None
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

        self._add_standard_train_hooks()

        with tf.device(self._base_device):
            self._create_optimizer_trainable_vars()
        
        