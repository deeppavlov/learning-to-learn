import tensorflow as tf
from meta import Meta
from useful_functions import block_diagonal


class ResNet4Lstm(Meta):

    def _create_optimizer_states(self, num_exercises, gpu_idx):
        with tf.variable_scope('optimizer_states_on_gpu_%s' % gpu_idx):
            states = [
                tf.get_variable(
                    'h', tf.zeros([num_exercises, self._num_lstm_nodes]), trainable=False),
                tf.get_variable(
                    'c', tf.zeros([num_exercises, self._num_lstm_nodes]), trainable=False)
            ]
            return states

    @staticmethod
    def _reset_optimizer_states(gpu_idx):
        with tf.variable_scope('optimizer_states_on_gpu_%s' % gpu_idx, resue=True):
            h = tf.get_variable('h')
            c = tf.get_variable('c')
            h_shape = h.get_shape.as_list()
            c_shape = c.get_shape().as_list()
            reset_ops = [
                tf.assign(h, tf.zeros(h_shape)),
                tf.assign(c, tf.zeros(c_shape))
            ]
            return tf.group(*reset_ops)

    @staticmethod
    def _create_permutation_matrix(size, num_exercises):
        return tf.one_hot(
            tf.stack(
                [tf.random_shuffle([i for i in range(size)])
                 for _ in range(num_exercises)]),
            size)

    def _reset_permutations(self, gpu_idx):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='permutation_matrices_on_gpu_%s' % gpu_idx)
        reset_ops = list()
        for v in variables:
            v_shape = v.get_shape().as_list()
            reset_ops.append(
                tf.assign(v, self._create_permutation_matrix(v_shape[1], v_shape[0]))
            )
        return tf.group(*reset_ops)

    def _extend_with_permutations(self, optimizer_ins, num_exrcises, gpu_idx):
        net_size = self._pupil.get_net_size()
        num_nodes = net_size['num_nodes']
        num_output_nodes = net_size['num_output_nodes']
        num_layers = len(num_nodes)
        num_output_layers = len(num_output_nodes)
        with tf.variable_scope('permutation_matrices_on_gpu_%s' % gpu_idx):
            if 'embedding_layer' in optimizer_ins:
                emb = tf.get_variable(
                    'embedding',
                    self._create_permutation_matrix(net_size['embedding_size'], num_exrcises),
                    trainable=False
                )
            lstm_layers = list()
            for layer_idx in range(num_layers):
                lstm_layers.append(tf.get_variable(
                    'c_%s' % layer_idx,
                    self._create_permutation_matrix(num_nodes[layer_idx], num_exrcises),
                    trainable=False
                ))
            output_layers = list()
            for layer_idx in range(num_output_layers-1):
                output_layers.append(tf.get_variable(
                    'h_%s' % layer_idx,
                    self._create_permutation_matrix(num_output_nodes[layer_idx], num_exrcises),
                    trainable=False
                ))
        if 'embedding_layer' in optimizer_ins:
            optimizer_ins['embedding_layer']['out_perm'] = emb
            optimizer_ins['lstm_layer_0']['in_perm'] = block_diagonal([emb, lstm_layers[0]])
        for layer_idx, c in enumerate(lstm_layers):
            optimizer_ins['lstm_layer_%s' % layer_idx]['out_perm'] = block_diagonal(
                [c] * 4
            )
            if layer_idx < num_layers - 1:
                optimizer_ins['lstm_layer_%s' % (layer_idx+1)]['in_perm'] = block_diagonal(
                    [c, lstm_layers[layer_idx+1]])
        optimizer_ins['output_layer_0']['in_perm'] = lstm_layers[-1]
        for layer_idx, h in output_layers:
            optimizer_ins['output_layer_%s' % layer_idx]['out_perm'] = h
            optimizer_ins['output_layer_%s' % (layer_idx+1)]['out_perm'] = output_layers[layer_idx+1]
        return optimizer_ins

    def _optimizer_core(self, optimizer_ins, num_exrcises, states, gpu_idx):
        pass

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

        _ = self._create_optimizer_states(False)

        if regime == 'train':
            ex_per_gpu = self._num_exercises // self._num_gpus
            remaining = self._num_exercises - self._num_gpus * ex_per_gpu
            self._exercise_gpu_map = [n // ex_per_gpu for n in range((self._num_gpus - 1) * ex_per_gpu)] + \
                                     [self._num_gpus - 1] * (ex_per_gpu + remaining)
            self._num_ex_on_gpus = [ex_per_gpu] * (self._num_gpus - 1) + [ex_per_gpu + remaining]
            self._gpu_borders = self._gpu_idx_borders(self._exercise_gpu_map)

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
        
        