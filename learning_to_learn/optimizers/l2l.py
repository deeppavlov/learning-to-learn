from itertools import chain
import numpy as np
import tensorflow as tf

from learning_to_learn.optimizers.meta import Meta
from learning_to_learn.useful_functions import flatten, construct_dict_without_none_entries, custom_matmul, \
    custom_add, construct
from learning_to_learn.tensors import log_and_sign

P = 10


class L2L(Meta):

    @staticmethod
    def check_kwargs(**kwargs):
        pass

    def _create_optimizer_states(self, num_exercises, var_scope, gpu_idx):
        states = dict()
        with tf.variable_scope(var_scope):
            with tf.variable_scope('gpu_%s' % gpu_idx):
                for layer_name, layer_sizes in self._opt_in_sizes.items():
                    with tf.variable_scope(layer_name):
                        states[layer_name] = dict()
                        d = states[layer_name]
                        nu = layer_sizes['nu']
                        if nu is None:
                            nu = 1
                        bs = layer_sizes['bs']
                        for inp_name in self._selected:
                            if self._opt_in_types[inp_name] == 'variable':
                                second_dim = layer_sizes[inp_name]
                            else:
                                second_dim = layer_sizes[inp_name] * bs * nu
                            with tf.variable_scope(inp_name):
                                d[inp_name] = []
                                for opt_layer_idx in range(self._num_lstm_layers):
                                    with tf.variable_scope('opt_lstm_layer_%s' % opt_layer_idx):
                                        d[inp_name].append(
                                            (
                                                tf.get_variable(
                                                    'h',
                                                    shape=[
                                                        num_exercises, second_dim, self._num_lstm_nodes[opt_layer_idx]],
                                                    initializer=tf.zeros_initializer(),
                                                    trainable=False,
                                                    collections=['OPT_STATES', tf.GraphKeys.GLOBAL_VARIABLES],
                                                ),
                                                tf.get_variable(
                                                    'c',
                                                    shape=[
                                                        num_exercises, second_dim, self._num_lstm_nodes[opt_layer_idx]],
                                                    initializer=tf.zeros_initializer(),
                                                    trainable=False,
                                                    collections=['OPT_STATES', tf.GraphKeys.GLOBAL_VARIABLES],
                                                ),
                                            )
                                        )
        return states

    @staticmethod
    def _reset_optimizer_states(var_scope, gpu_idx):
        with tf.variable_scope(var_scope, reuse=True):
            with tf.variable_scope('gpu_%s' % gpu_idx):
                return [
                    tf.variables_initializer(
                        tf.get_collection('OPT_STATES', '/'.join([var_scope, 'gpu_%s' % gpu_idx])),
                        name='reset_optimizer_states_initializer_on_gpu_%s' % gpu_idx
                    )
                ]

    def _reset_permutations(self, gpu_idx):
        return []

    def _create_permutation_matrices(self, num_exercises, gpu_idx):
        pass

    def _reset_all_permutation_matrices(self):
        return []

    def _compute_lstm_matrix_parameters(self, idx):
        if idx == 0:
            # print(self._num_nodes)
            input_dim = self._num_lstm_nodes[0] + 2
        else:
            input_dim = self._num_lstm_nodes[idx - 1] + self._num_lstm_nodes[idx]
        output_dim = 4 * self._num_lstm_nodes[idx]
        stddev = self._optimizer_init_parameter * np.sqrt(1. / (input_dim + output_dim))
        return input_dim, output_dim, stddev

    def _create_optimizer_trainable_vars(self):
        vars = dict()
        with tf.variable_scope('optimizer_trainable_variables'):
            for layer_name, layer_sizes in self._opt_in_sizes.items():
                with tf.variable_scope(layer_name):
                    vars[layer_name] = dict()
                    d = vars[layer_name]
                    for inp_name in self._selected:
                        with tf.variable_scope(inp_name):
                            d[inp_name] = dict(
                                lstm_matrices=list(),
                                lstm_biases=list(),
                                linear=None
                            )
                            for opt_layer_idx in range(self._num_lstm_layers):
                                in_dim, out_dim, stddev = self._compute_lstm_matrix_parameters(opt_layer_idx)
                                with tf.variable_scope('opt_lstm_layer_%s' % opt_layer_idx):
                                    d[inp_name]['lstm_matrices'].append(
                                        tf.get_variable(
                                            'lstm_matrix_%s' % opt_layer_idx,
                                            shape=[in_dim, out_dim],
                                            initializer=tf.truncated_normal_initializer(stddev=stddev),
                                            trainable=True,
                                            collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
                                        )
                                    )
                                    d[inp_name]['lstm_biases'].append(
                                        tf.get_variable(
                                            'lstm_bias_%s' % opt_layer_idx,
                                            shape=[out_dim],
                                            initializer=tf.zeros_initializer(),
                                            trainable=True,
                                            collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES],
                                        )
                                    )
                            stddev = self._optimizer_init_parameter * np.sqrt(1. / (self._num_lstm_nodes[-1] + 1))
                            d[inp_name]['linear'] = tf.get_variable(
                                'linear',
                                shape=[self._num_lstm_nodes[-1], 1],
                                initializer=tf.truncated_normal_initializer(stddev=stddev),
                                trainable=True,
                                collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
                            )
        return vars

    # def _optimizer_core(self, optimizer_ins, num_exercises, states, gpu_idx):
    #     # optimizer_ins = self._extend_with_permutations(optimizer_ins, num_exercises, gpu_idx)
    #     # optimizer_ins = self._forward_permute(optimizer_ins)
    #     return self._empty_core(optimizer_ins)

    def _apply_lstm_layer(self, inp, state, matrix, bias, scope='lstm'):
        with tf.name_scope(scope):
            x = tf.concat(
                [tf.nn.dropout(
                    inp,
                    self._optimizer_dropout_keep_prob),
                 state[0]],
                -1,
                name='X')
            s = custom_matmul(x, matrix)
            linear_res = custom_add(
                s, bias, name='linear_res')
            state_dim = tf.shape(state[0])[-1:]
            split_dims = tf.concat([3 * state_dim, state_dim], 0)
            [sigm_arg, tanh_arg] = tf.split(linear_res, split_dims, axis=-1, name='split_to_act_func_args')
            sigm_res = tf.sigmoid(sigm_arg, name='sigm_res')
            transform_vec = tf.tanh(tanh_arg, name='transformation_vector')
            [forget_gate, input_gate, output_gate] = tf.split(sigm_res, 3, axis=-1, name='gates')
            new_cell_state = tf.add(forget_gate * state[1], input_gate * transform_vec, name='new_cell_state')
            new_hidden_state = tf.multiply(output_gate, tf.tanh(new_cell_state), name='new_hidden_state')
        return new_hidden_state, new_cell_state

    @staticmethod
    def _opt_in_reshaping(inp):
        with tf.name_scope('reshape_opt_in'):
            if isinstance(inp, list):
                inp = tf.stack(inp, 1)
                stack = True
            else:
                stack = False
            sh = tf.shape(inp)
            new_sh = tf.concat([sh[:1], tf.constant([-1], dtype=tf.int32)], 0, name='new_shape')
            return tf.reshape(inp, new_sh), sh, stack

    @staticmethod
    def _reshape_back(inp, old_shape, unstack):
        with tf.name_scope('reshape_back_opt_ins'):
            inp = tf.squeeze(inp, axis=-1)
            inp = tf.reshape(inp, old_shape)
            if unstack:
                res = tf.unstack(inp, axis=1)
            else:
                res = inp
        return res

    def _apply_net(
            self,
            inp,
            state,
            vars,
    ):
        inp, old_shape, stack = self._opt_in_reshaping(inp)
        inp = log_and_sign(inp, P)
        new_state = list()
        for layer_idx, (layer_state, matrix, bias) in enumerate(zip(state, vars['lstm_matrices'], vars['lstm_biases'])):
            new_h, new_c = self._apply_lstm_layer(
                inp,
                layer_state,
                matrix,
                bias,
                scope='lstm_layer_%s' % layer_idx,
            )
            new_state.append(
                (new_h, new_c)
            )
            inp = new_h
        with tf.name_scope('linear'):
            linear_res = self._scale * custom_matmul(inp, vars['linear'])
        res = self._reshape_back(linear_res, old_shape, stack)
        return res, new_state

    def _optimizer_core(self, optimizer_ins, state, gpu_idx, permute=True):
        new_state = construct(state)
        # print(state)
        for ok, ov in optimizer_ins.items():
            with tf.name_scope(ok):
                for inp_name in self._selected:
                    if inp_name in ov:
                        with tf.name_scope(inp_name):
                            ov[inp_name + '_pr'], new_state[ok][inp_name] = self._apply_net(
                                ov[inp_name],
                                state[ok][inp_name],
                                self._opt_trainable[ok][inp_name]
                            )
        # for ok, ov in optimizer_ins.items():
        #     for ik, iv in ov.items():
        #         print()
        #         print(ok, ik)
        #         print(iv)
        return optimizer_ins, new_state

    def __init__(
            self,
            pupil,
            num_exercises=10,
            num_optimizer_unrollings=10,
            num_gpus=1,
            num_lstm_layers=2,
            num_lstm_nodes=None,
            regularization_rate=1e-7,
            optimizer_init_parameter=.1,
            regime='train',
            optimizer_for_opt_type='adam',
            selected=None,
            additional_metrics=None,
            flags=None,
            get_theta=False,
            get_omega_and_beta=True,
            matrix_mod='omega',
            scale=0.01,
            no_end=True,
    ):
        if additional_metrics is None:
            additional_metrics = list()
        if flags is None:
            flags = list()
        if selected is None:
            selected = ['omega', 'beta']
        if num_lstm_nodes is None:
            num_lstm_nodes = [20, 20]

        self._pupil = pupil
        self._pupil_net_size = self._pupil.get_net_size()
        self._pupil_dims = self._pupil.get_layer_dims()
        self._opt_in_sizes = self._get_opt_in_sizes(self._pupil_dims, self._pupil_net_size)
        self._emb_layer_is_present = 'embedding_size' in self._pupil_net_size
        self._num_exercises = num_exercises
        self._num_optimizer_unrollings = num_optimizer_unrollings
        self._num_lstm_nodes = num_lstm_nodes
        self._num_lstm_layers = num_lstm_layers
        self._num_gpus = num_gpus
        if self._num_gpus == 1:
            self._base_device = '/gpu:0'
        else:
            self._base_device = '/cpu:0'

        self._selected = selected

        self._regularization_rate = regularization_rate
        self._inp_gradient_clipping = None
        self._optimizer_init_parameter = optimizer_init_parameter
        self._regime = regime
        self._scale = scale

        self._optimizer_for_opt_type = optimizer_for_opt_type

        self._additional_metrics = additional_metrics

        self._matrix_mod = matrix_mod
        self._permute = False
        self._flags = flags
        self._get_theta = get_theta
        self._get_omega_and_beta = get_omega_and_beta
        self._no_end = no_end

        self._normalizing = None
        self._hooks = dict(
            pupil_grad_eval_inputs=None,
            pupil_grad_eval_labels=None,
            optimizer_grad_inputs=None,
            optimizer_grad_labels=None,
            pupil_savers=None,
            optimizer_train_op=None,
            learning_rate_for_optimizer_training=None,
            train_with_meta_optimizer_op=None,
            reset_optimizer_train_state=None,
            reset_optimizer_inference_state=None,
            reset_permutation_matrices=None,
            reset_pupil_grad_eval_pupil_storage=None,
            reset_optimizer_grad_pupil_storage=None,
            reset_optimizer_inference_pupil_storage=None,
            meta_optimizer_saver=None,
            loss=None,
            start_loss=None,
            end_loss=None,
            optimizer_dropout_keep_prob=None,
            pupil_trainable_initializers=None,
            train_optimizer_summary=None
        )
        for add_metric in self._additional_metrics:
            self._hooks['start_' + add_metric] = None
            self._hooks['end_' + add_metric] = None
            self._hooks[add_metric] = None

        self._debug_tensors = list()

        self._optimizer_dropout_keep_prob = tf.placeholder(tf.float32, name='optimizer_dropout_keep_prob')
        self._hooks['optimizer_dropout_keep_prob'] = self._optimizer_dropout_keep_prob
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
                self._pupil_savers, self._pupil_trainable_initializers = self._create_pupil_variables_and_savers(
                    self._pupil, self._num_exercises, self._exercise_gpu_map)
            self._hooks['pupil_savers'] = self._pupil_savers
            self._hooks['pupil_trainable_initializers'] = self._pupil_trainable_initializers
            self._hooks['reset_pupil_grad_eval_pupil_storage'] = tf.group(
                *chain(*[self._pupil.reset_storage(stor) for stor in self._pupil_grad_eval_pupil_storage])
            )
            self._hooks['reset_optimizer_grad_pupil_storage'] = tf.group(
                *chain(*[self._pupil.reset_storage(stor) for stor in self._optimizer_grad_pupil_storage])
            )
            self._hooks['reset_optimizer_inference_pupil_storage'] = tf.group(*self._pupil.reset_self_train_storage())
            self._add_standard_train_hooks()

            self._additional_loss = 0
        else:
            self._exercise_gpu_map = None
            self._pupil_grad_eval_inputs, self._pupil_grad_eval_labels, \
                self._optimizer_grad_inputs, self._optimizer_grad_labels = None, None, None, None

        with tf.device(self._base_device):
            self._opt_trainable = self._create_optimizer_trainable_vars()
            self._hooks['meta_optimizer_saver'] = self.create_saver()

        if self._regime == 'train':
            self._learning_rate_for_optimizer_training = tf.placeholder(
                tf.float32, name='learning_rate_for_optimizer_training')
            self._hooks['learning_rate_for_optimizer_training'] = self._learning_rate_for_optimizer_training
            if self._optimizer_for_opt_type == 'adam':
                self._optimizer_for_optimizer_training = tf.train.AdamOptimizer(
                    learning_rate=self._learning_rate_for_optimizer_training)
            elif self._optimizer_for_opt_type == 'sgd':
                self._optimizer_for_optimizer_training = tf.train.GradientDescentOptimizer(
                    learning_rate=self._learning_rate_for_optimizer_training)
            self._train_graph()
            self._inference_graph()
            self._empty_op = tf.constant(0)
            self._hooks['reset_optimizer_train_state'] = self._empty_op
            self._hooks['reset_optimizer_inference_state'] = self._empty_op
            self._hooks['reset_permutation_matrices'] = self._empty_op
        elif self._regime == 'inference':
            self._inference_graph()
            self._hooks['reset_optimizer_inference_state'] = self._empty_op

    def get_default_hooks(self):
        return construct_dict_without_none_entries(self._hooks)

    def create_saver(self):
        # print("(Lstm.create_saver)var_dict:", var_dict)
        with tf.device('/cpu:0'):
            saved_vars = dict()
            for ok, ov in self._opt_trainable.items():
                for ik, iv in ov.items():
                    for idx, (m, b) in enumerate(zip(iv['lstm_matrices'], iv['lstm_biases'])):
                        saved_vars['%s/%s/lstm_matrix_%s' % (ok, ik, idx)] = m
                        saved_vars['%s/%s/lstm_bias_%s' % (ok, ik, idx)] = b
                    saved_vars['%s/%s/linear' % (ok, ik)] = iv['linear']
            saver = tf.train.Saver(saved_vars, max_to_keep=None)
        return saver
