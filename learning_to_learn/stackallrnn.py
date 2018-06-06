from itertools import chain

import tensorflow as tf
from learning_to_learn.useful_functions import flatten, construct_dict_without_none_entries, construct,\
    unite_nested_dicts, block_diagonal, custom_matmul, custom_add

from learning_to_learn.meta import Meta


class StackAllRnn(Meta):

    @staticmethod
    def form_kwargs(kwargs_for_building, insertions):
        for insertion in insertions:
            if insertion['list_index'] is None:
                kwargs_for_building[insertion['hp_name']] = insertion['paste']
            else:
                kwargs_for_building[insertion['hp_name']][insertion['list_index']] = insertion['paste']
        return kwargs_for_building

    @staticmethod
    def check_kwargs(**kwargs):
        pass

    def _create_optimizer_states(self, num_exercises, var_scope, gpu_idx):
        with tf.variable_scope(var_scope):
            with tf.variable_scope('gpu_%s' % gpu_idx):
                states = [
                    tf.get_variable(
                        'h',
                        shape=[num_exercises, self._pupil_net_size['batch_size'], self._total_ndim],
                        initializer=tf.zeros_initializer(),
                        trainable=False),
                    tf.get_variable(
                        'c',
                        shape=[num_exercises, self._pupil_net_size['batch_size'], self._total_ndim],
                        initializer=tf.zeros_initializer(),
                        trainable=False)
                ]
                return states

    @staticmethod
    def _reset_optimizer_states(var_scope, gpu_idx):
        with tf.variable_scope(var_scope, reuse=True):
            with tf.variable_scope('gpu_%s' % gpu_idx):
                h = tf.get_variable('h')
                c = tf.get_variable('c')
                # print("(ResNet4Lstm._reset_optimizer_states)h:", h)
                # print("(ResNet4Lstm._reset_optimizer_states)c:", c)
                return [tf.variables_initializer([h, c], name='reset_optimizer_states_initializer_on_gpu_%s' % gpu_idx)]

    @staticmethod
    def _create_permutation_matrix(size, num_exercises):
        with tf.device('/cpu:0'):
            map_ = tf.stack(
                [tf.random_shuffle([i for i in range(size)])
                 for _ in range(num_exercises)])
        return tf.one_hot(map_, size)

    def _reset_permutations(self, gpu_idx):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='permutation_matrices_on_gpu_%s' % gpu_idx)
        reset_ops = list()
        for v in variables:
            v_shape = v.get_shape().as_list()
            reset_ops.append(
                tf.assign(v, self._create_permutation_matrix(v_shape[1], v_shape[0]))
            )
        return reset_ops

    def _create_permutation_matrices(self, num_exercises, gpu_idx):
        num_nodes = self._pupil_net_size['num_nodes']
        num_output_nodes = self._pupil_net_size['num_output_nodes']
        with tf.variable_scope('permutation_matrices_on_gpu_%s' % gpu_idx):
            if self._emb_layer_is_present:
                _ = tf.get_variable(
                    'embedding',
                    initializer=self._create_permutation_matrix(self._pupil_net_size['embedding_size'], num_exercises),
                    trainable=False
                )
            for layer_idx in range(self._pupil_net_size['num_layers']):
                _ = tf.get_variable(
                    'c_%s' % layer_idx,
                    initializer=self._create_permutation_matrix(num_nodes[layer_idx], num_exercises),
                    trainable=False
                )
            for layer_idx in range(self._pupil_net_size['num_output_layers'] - 1):
                _ = tf.get_variable(
                    'h_%s' % layer_idx,
                    initializer=self._create_permutation_matrix(num_output_nodes[layer_idx], num_exercises),
                    trainable=False
                )

    def _reset_all_permutation_matrices(self):
        reset_ops = list()
        for gpu_idx, num_exercises in enumerate(self._num_ex_on_gpus):
            reset_ops.extend(self._reset_permutations(gpu_idx))
        return reset_ops

    def _extend_with_permutations(self, optimizer_ins, gpu_idx):
        num_layers = self._pupil_net_size['num_layers']
        num_output_layers = self._pupil_net_size['num_output_layers']
        with tf.variable_scope('permutation_matrices_on_gpu_%s' % gpu_idx, reuse=True):
            if self._emb_layer_is_present:
                emb = tf.get_variable('embedding')
            lstm_layers = list()
            for layer_idx in range(num_layers):
                lstm_layers.append(tf.get_variable('c_%s' % layer_idx))
            output_layers = list()
            for layer_idx in range(num_output_layers-1):
                output_layers.append(tf.get_variable('h_%s' % layer_idx))
        if self._emb_layer_is_present:
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

    def _create_optimizer_trainable_vars(self):
        stddev = self._optimizer_init_parameter / (6 * self._total_ndim) ** .5
        with tf.variable_scope('optimizer_trainable_variables'):
            matrix = tf.Variable(
                tf.truncated_normal(
                    [2*self._total_ndim, 4*self._total_ndim],
                    stddev=stddev
                ),
                name='matrix'
            )
            bias = tf.Variable(tf.zeros([4*self._total_ndim]), name='bias')
            vars = [matrix, bias]
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, matrix)
        return vars

    # def _optimizer_core(self, optimizer_ins, num_exercises, states, gpu_idx):
    #     # optimizer_ins = self._extend_with_permutations(optimizer_ins, num_exercises, gpu_idx)
    #     # optimizer_ins = self._forward_permute(optimizer_ins)
    #     return self._empty_core(optimizer_ins)

    def _apply_lstm_layer(self, inp, state, matrix, bias, scope='lstm'):
        with tf.name_scope(scope):
            nn = self._total_ndim
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
            [sigm_arg, tanh_arg] = tf.split(linear_res, [3 * nn, nn], axis=-1, name='split_to_act_func_args')
            sigm_res = tf.sigmoid(sigm_arg, name='sigm_res')
            transform_vec = tf.tanh(tanh_arg, name='transformation_vector')
            [forget_gate, input_gate, output_gate] = tf.split(sigm_res, 3, axis=-1, name='gates')
            new_cell_state = tf.add(forget_gate * state[1], input_gate * transform_vec, name='new_cell_state')
            new_hidden_state = tf.multiply(output_gate, tf.tanh(new_cell_state), name='new_hidden_state')
        return new_hidden_state, new_cell_state


    def _optimizer_core(self, optimizer_ins, state, gpu_idx, permute=True):
        if permute:
            optimizer_ins = self._extend_with_permutations(optimizer_ins, gpu_idx)
            optimizer_ins = self._forward_permute(optimizer_ins, ['o'], ['sigma'])
        vec, map_ = self._all_ins_2_1_vec(optimizer_ins, ['o', 'sigma'])
        state = self._apply_lstm_layer(vec, state, self._opt_trainable[0], self._opt_trainable[1])
        outs = self._unpack_all_ins_from_1_vec(state[0], map_)
        outs = self._mv_tensors(outs, ['o', 'sigma'], ['o_pr', 'sigma_pr'])
        optimizer_outs = unite_nested_dicts([optimizer_ins, outs], 1)
        if permute:
            optimizer_outs = self._backward_permute(optimizer_outs, ['o_pr'], ['sigma_pr'])
        return optimizer_outs, state

    def __init__(
            self,
            pupil,
            num_exercises=10,
            num_optimizer_unrollings=10,
            perm_period=None,
            num_gpus=1,
            regularization_rate=1e-7,
            clip_norm=1e+5,
            optimizer_init_parameter=.1,
            regime='train',
            optimizer_for_opt_type='adam',
            additional_metrics=None,
            flags=None,
            normalizing=None,
            permute=False
    ):
        if additional_metrics is None:
            additional_metrics = list()
        if flags is None:
            flags = list()

        self._pupil = pupil
        self._pupil_net_size = self._pupil.get_net_size()
        self._pupil_dims = self._pupil.get_layer_dims()
        self._emb_layer_is_present = 'embedding_size' in self._pupil_net_size
        self._num_exercises = num_exercises
        self._num_optimizer_unrollings = num_optimizer_unrollings
        self._perm_period = perm_period
        self._num_gpus = num_gpus
        if self._num_gpus == 1:
            self._base_device = '/gpu:0'
        else:
            self._base_device = '/cpu:0'
        self._regularization_rate = regularization_rate
        self._clip_norm = clip_norm
        self._optimizer_init_parameter = optimizer_init_parameter
        self._regime = regime

        self._optimizer_for_opt_type = optimizer_for_opt_type

        self._additional_metrics = additional_metrics

        self._flags = flags

        self._normalizing = normalizing

        self._permute = permute
        self._total_ndim = sum(flatten(self._pupil_dims)) * self._pupil_net_size['num_unrollings']
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
