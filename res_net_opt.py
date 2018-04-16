import tensorflow as tf
from meta import Meta
from useful_functions import (block_diagonal, custom_matmul, custom_add, flatten,
                              construct_dict_without_none_entries, print_optimizer_ins)


class ResNet4Lstm(Meta):

    @staticmethod
    def check_kwargs(**kwargs):
        pass

    def _create_optimizer_states(self, num_exercises, var_scope, gpu_idx):
        with tf.variable_scope(var_scope):
            with tf.variable_scope('gpu_%s' % gpu_idx):
                states = [
                    tf.get_variable(
                        'h',
                        shape=[num_exercises, self._num_lstm_nodes],
                        initializer=tf.zeros_initializer(),
                        trainable=False),
                    tf.get_variable(
                        'c',
                        shape=[num_exercises, self._num_lstm_nodes],
                        initializer=tf.zeros_initializer(),
                        trainable=False)
                ]
                return states

    # def _create_optimizer_states(self, num_exercises, var_scope, gpu_idx):
    #     return []

    @staticmethod
    def _reset_optimizer_states(var_scope, gpu_idx):
        with tf.variable_scope(var_scope, reuse=True):
            with tf.variable_scope('gpu_%s' % gpu_idx):
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
    def _reset_optimizer_states(var_scope, gpu_idx):
        return None

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

    @staticmethod
    def _distribute_rnn_size_among_res_layers(rnn_size, num_res_layers):
        rnn_nodes_for_layer = rnn_size // num_res_layers
        remainder = rnn_size % num_res_layers
        return [rnn_nodes_for_layer for _ in range(num_res_layers-1)] + [rnn_nodes_for_layer + remainder]

    @staticmethod
    def _res_core_vars(left, target, right, rnn_part_size, res_size, var_scope):
        with tf.variable_scope(var_scope):
            matrices, biases = list(), list()
            in_ndims = sum(flatten(left)) + sum(flatten(target)) + sum(flatten(right)) + rnn_part_size
            out_ndims = sum(flatten(target)) + rnn_part_size
            stddev = 2. / (in_ndims + res_size)**.5
            with tf.variable_scope('in_core'):
                matrices.append(
                    tf.get_variable(
                        'matrix',
                        shape=[in_ndims, res_size],
                        initializer=tf.truncated_normal_initializer(stddev=stddev)
                    )
                )
                biases.append(
                    tf.get_variable(
                        'bias',
                        shape=[res_size],
                        initializer=tf.zeros_initializer()
                    )
                )
            with tf.variable_scope('out_core'):
                matrices.append(
                    tf.get_variable(
                        'matrix',
                        shape=[res_size, out_ndims],
                        initializer=tf.truncated_normal_initializer(stddev=stddev)
                    )
                )
                biases.append(
                    tf.get_variable(
                        'bias',
                        shape=[out_ndims],
                        initializer=tf.zeros_initializer()
                    )
                )
            for m in matrices:
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, m)
        return matrices, biases

    def _create_optimizer_trainable_vars(self):
        if self._emb_layer_is_present:
            embedding_layer = self._pupil_dims['embedding_layer']
        lstm_layers = self._pupil_dims['lstm_layers']
        output_layers = self._pupil_dims['output_layers']
        num_layers = self._pupil_net_size['num_layers']
        num_output_layers = self._pupil_net_size['num_output_layers']
        vars = dict()
        with tf.variable_scope('optimizer_trainable_variables'):
            with tf.variable_scope('res_layers'):
                vars['res_layers'] = list()
                for res_idx, rnn_part in enumerate(self._rnn_for_res_layers):
                    with tf.variable_scope('layer_%s' % res_idx):
                        if self._emb_layer_is_present:
                            res_layer_params = dict()
                            res_layer_params['embedding_layer'] = self._res_core_vars(
                                [()],
                                [embedding_layer],
                                [lstm_layers[0]],
                                rnn_part,
                                self._res_size,
                                'embedding_layer_core'
                            )
                        for layer_idx, layer_dims in enumerate(lstm_layers):
                            if layer_idx == 0:
                                if self._emb_layer_is_present:
                                    pupil_previous_layer_dims = embedding_layer
                                else:
                                    pupil_previous_layer_dims = ()
                            else:
                                pupil_previous_layer_dims = lstm_layers[layer_idx-1]
                            if layer_idx == num_layers - 1:
                                pupil_next_layer_dims = output_layers[0]
                            else:
                                pupil_next_layer_dims = lstm_layers[layer_idx+1]
                            res_layer_params['lstm_layer_%s' % layer_idx] = self._res_core_vars(
                                [pupil_previous_layer_dims,
                                 layer_dims],
                                [layer_dims],
                                [pupil_next_layer_dims, layer_dims],
                                rnn_part,
                                self._res_size,
                                'lstm_layer_%s_core' % layer_idx
                            )
                        for layer_idx, layer_dims in enumerate(output_layers):
                            if layer_idx == 0:
                                pupil_previous_layer_dims = lstm_layers[-1]
                            else:
                                pupil_previous_layer_dims = output_layers[layer_idx-1]
                            if layer_idx == num_output_layers - 1:
                                pupil_next_layer_dims = ()
                            else:
                                pupil_next_layer_dims = output_layers[layer_idx+1]
                            res_layer_params['output_layer_%s' % layer_idx] = self._res_core_vars(
                                [pupil_previous_layer_dims],
                                [layer_dims],
                                [pupil_next_layer_dims],
                                rnn_part,
                                self._res_size,
                                'output_layer_%s_core' % layer_idx
                            )
                        vars['res_layers'].append(res_layer_params)
                with tf.variable_scope('lstm'):
                    rnn_size = self._num_lstm_nodes
                    stddev = 2. / (6 * rnn_size) ** .5
                    vars['lstm_matrix'] = tf.get_variable(
                        'lstm_matrix',
                        shape=[2 * rnn_size, 4 * rnn_size],
                        initializer=tf.truncated_normal_initializer(stddev=stddev)
                    )
                    vars['lstm_bias'] = tf.get_variable(
                        'lstm_bias',
                        shape=[4 * rnn_size],
                        initializer=tf.zeros_initializer()
                    )
        return vars

    # def _optimizer_core(self, optimizer_ins, num_exercises, states, gpu_idx):
    #     # optimizer_ins = self._extend_with_permutations(optimizer_ins, num_exercises, gpu_idx)
    #     # optimizer_ins = self._forward_permute(optimizer_ins)
    #     return self._empty_core(optimizer_ins)

    def _pad(self, tensor, direction):
        """if direction < 0 |direction| most recent pupil unrollings are cut and  paddings are added after oldest
        unrollings. If direction > 0 all is done the opposite way."""
        pupil_batch_size = self._pupil_net_size['batch_size']
        tensor_shape = tensor.get_shape().as_list()
        padded_size = pupil_batch_size * abs(direction)
        kept_size = tensor_shape[-2] - pupil_batch_size * abs(direction)
        if direction < 0:
            split_sizes = [kept_size, padded_size]
            kept_idx = 0
        else:
            split_sizes = [padded_size, kept_size]
            kept_idx = 1
        kept = tf.split(tensor, split_sizes, axis=-2)[kept_idx]
        paddings = tf.zeros(tensor_shape[:-2] + [padded_size] + tensor_shape[-1:])
        if direction < 0:
            padded = tf.concat([paddings, kept], -2)
        else:
            padded = tf.concat([kept, paddings], -2)
        return padded

    @staticmethod
    def _apply_res_core(vars, opt_ins, rnn_part, scope, target_dims):
        with tf.name_scope(scope):
            # print("\n(ResNet4Lstm._apply_res_core)rnn_part:", rnn_part)
            # print('(ResNet4Lstm._apply_res_core)opt_ins:', opt_ins)
            opt_ins_united = tf.concat(opt_ins, -1)
            rnn_stack_num = opt_ins_united.get_shape().as_list()[-2]
            rnn_part = tf.stack([rnn_part]*rnn_stack_num, axis=-2)
            # print("(ResNet4Lstm._apply_res_core)rnn_part:", rnn_part)
            hs = tf.concat([opt_ins_united, rnn_part], -1)
            matrices = vars[0]
            biases = vars[1]
            for m, b in zip(matrices, biases):
                # print('\n(ResNet4Lstm._apply_res_core)hs:', hs)
                # print('(ResNet4Lstm._apply_res_core)m:', m)
                hs = tf.nn.relu(custom_add(custom_matmul(hs, m), b))
            rnn_part_dim = hs.get_shape().as_list()[-1] - sum(target_dims)
            o, sigma, rnn_part = tf.split(hs, list(target_dims) + [rnn_part_dim], axis=-1)
            # print("(ResNet4Lstm._apply_res_core)rnn_part:", rnn_part)
            return o, sigma, rnn_part

    def _apply_res_layer(self, ins, res_vars, rnn_part, scope):
        with tf.name_scope(scope):
            if self._emb_layer_is_present:
                core_inps = [
                    ins['embedding_layer']['o_c'],
                    ins['embedding_layer']['sigma_c'],
                    ins['lstm_layer_0']['o_c'],
                    ins['lstm_layer_0']['sigma_c']
                ]
                # print('(ResNet4Lstm._apply_res_layer)core_inps:', core_inps)
                # print('(ResNet4Lstm._apply_res_layer)res_vars:', res_vars)
                o, sigma, emb_rnn_part = self._apply_res_core(
                    res_vars['embedding_layer'], core_inps, rnn_part,
                    'embedding_layer', self._pupil_dims['embedding_layer'])
                ins['embedding_layer']['o_c'] = o
                ins['embedding_layer']['sigma_c'] = sigma
            else:
                emb_rnn_part = 0

            lstm_rnn_parts = list()
            for layer_idx in range(self._pupil_net_size['num_layers']):
                if layer_idx == 0:
                    if self._emb_layer_is_present:
                        previous_layer_tensors = [
                            ins['embedding_layer']['o_c'],
                            ins['embedding_layer']['sigma_c']
                        ]
                    else:
                        previous_layer_tensors = []
                else:
                    previous_layer_tensors = [
                        ins['lstm_layer_%s' % (layer_idx - 1)]['o_c'],
                        ins['lstm_layer_%s' % (layer_idx - 1)]['sigma_c']
                    ]
                if layer_idx == self._pupil_net_size['num_layers'] - 1:
                    next_layer_tensors = [
                        ins['output_layer_0']['o_c'],
                        ins['output_layer_0']['sigma_c']
                    ]
                else:
                    next_layer_tensors = [
                        ins['lstm_layer_%s' % (layer_idx + 1)]['o_c'],
                        ins['lstm_layer_%s' % (layer_idx + 1)]['sigma_c']
                    ]
                core_inps = [
                    *previous_layer_tensors,
                    ins['lstm_layer_%s' % layer_idx]['o_c'],
                    ins['lstm_layer_%s' % layer_idx]['sigma_c'],
                    self._pad(ins['lstm_layer_%s' % layer_idx]['o_c'], -1),
                    self._pad(ins['lstm_layer_%s' % layer_idx]['sigma_c'], -1),
                    self._pad(ins['lstm_layer_%s' % layer_idx]['o_c'], 1),
                    self._pad(ins['lstm_layer_%s' % layer_idx]['sigma_c'], 1),
                    *next_layer_tensors
                ]
                layer_name = 'lstm_layer_%s' % layer_idx
                o, sigma, lstm_rnn_part = self._apply_res_core(
                    res_vars[layer_name], core_inps, rnn_part, layer_name, self._pupil_dims['lstm_layers'][layer_idx])
                lstm_rnn_parts.append(lstm_rnn_part)
                ins['lstm_layer_%s' % layer_idx]['o_c'] = o
                ins['lstm_layer_%s' % layer_idx]['sigma_c'] = sigma

            output_rnn_parts = list()
            for layer_idx in range(self._pupil_net_size['num_output_layers']):
                if layer_idx == 0:
                    previous_layer_tensors = [
                        ins['lstm_layer_%s' % (self._pupil_net_size['num_layers'] - 1)]['o_c'],
                        ins['lstm_layer_%s' % (self._pupil_net_size['num_layers'] - 1)]['sigma_c']
                    ]
                else:
                    previous_layer_tensors = [
                        ins['output_layer_%s' % (layer_idx - 1)]['o_c'],
                        ins['output_layer_%s' % (layer_idx - 1)]['sigma_c']
                    ]
                if layer_idx == self._pupil_net_size['num_output_layers'] - 1:
                    next_layer_tensors = []
                else:
                    next_layer_tensors = [
                        ins['output_layer_%s' % (layer_idx + 1)]['o_c'],
                        ins['output_layer_%s' % (layer_idx + 1)]['sigma_c']
                    ]
                core_inps = [
                    *previous_layer_tensors,
                    ins['output_layer_%s' % layer_idx]['o_c'],
                    ins['output_layer_%s' % layer_idx]['sigma_c'],
                    *next_layer_tensors
                ]
                layer_name = 'output_layer_%s' % layer_idx
                # print("(ResNet4Lstm._apply_res_layer)layer_idx:", layer_idx)
                # print("(ResNet4Lstm._apply_res_layer)core_inps:", core_inps)
                # print("(ResNet4Lstm._apply_res_layer)rnn_part:", rnn_part)
                o, sigma, output_rnn_part = self._apply_res_core(
                    res_vars[layer_name], core_inps, rnn_part, layer_name, self._pupil_dims['output_layers'][layer_idx])
                output_rnn_parts.append(output_rnn_part)
                ins['output_layer_%s' % layer_idx]['o_c'] = o
                ins['output_layer_%s' % layer_idx]['sigma_c'] = sigma
            # print("(ResNet4Lstm._apply_res_layer)emb_rnn_part:", emb_rnn_part)
            # print("(ResNet4Lstm._apply_res_layer)lstm_rnn_parts:", lstm_rnn_parts)
            # print("(ResNet4Lstm._apply_res_layer)output_rnn_parts:", output_rnn_parts)
            return ins, emb_rnn_part + sum(lstm_rnn_parts + output_rnn_parts)

    def _apply_lstm_layer(self, inp, state, matrix, bias, scope='lstm'):
        with tf.name_scope(scope):
            nn = self._num_lstm_nodes
            x = tf.concat(
                [tf.nn.dropout(
                    inp,
                    self._optimizer_dropout_keep_prob),
                 state[0]],
                -1,
                name='X')
            s = tf.matmul(x, matrix)
            linear_res = tf.add(
                s, bias, name='linear_res')
            [sigm_arg, tanh_arg] = tf.split(linear_res, [3 * nn, nn], axis=-1, name='split_to_act_func_args')
            sigm_res = tf.sigmoid(sigm_arg, name='sigm_res')
            transform_vec = tf.tanh(tanh_arg, name='transformation_vector')
            [forget_gate, input_gate, output_gate] = tf.split(sigm_res, 3, axis=-1, name='gates')
            new_cell_state = tf.add(forget_gate * state[1], input_gate * transform_vec, name='new_cell_state')
            new_hidden_state = tf.multiply(output_gate, tf.tanh(new_cell_state), name='new_hidden_state')
        return new_hidden_state, new_cell_state

    def _optimizer_core(self, optimizer_ins, state, gpu_idx):
        optimizer_ins = self._extend_with_permutations(optimizer_ins, gpu_idx)
        # print('(ResNet4Lstm._optimizer_core)optimizer_ins\nBEFORE DIMS EXPANSION:')
        # print_optimizer_ins(optimizer_ins)
        # ndims = self._get_optimizer_ins_ndims(optimizer_ins)
        # if ndims == 2:
        #     optimizer_ins = self._expand_num_ex_dim_in_opt_ins(optimizer_ins, ['o', 'sigma'])
        # print('(ResNet4Lstm._optimizer_core)optimizer_ins\nBEFORE PERMUTATION:')
        # print_optimizer_ins(optimizer_ins)
        optimizer_ins = self._forward_permute(optimizer_ins, ['o'], ['sigma'])
        # print('(ResNet4Lstm._optimizer_core)optimizer_ins\nBEFORE CONCATENATION:')
        # print_optimizer_ins(optimizer_ins)
        optimizer_ins, num_concatenated = self._concat_opt_ins(optimizer_ins, ['o', 'sigma'])
        # print('(ResNet4Lstm._optimizer_core)optimizer_ins\nAFTER CONCATENATION:')
        # print_optimizer_ins(optimizer_ins)
        rnn_output_by_res_layers = tf.split(state[0], self._rnn_for_res_layers, axis=-1)
        # print("(ResNet4Lstm._optimizer_core)self._opt_trainable['res_layers']:", self._opt_trainable['res_layers'])
        rnn_input_by_res_layers = list()
        # print('(ResNet4Lstm._optimizer_core)rnn_output_by_res_layers:', rnn_output_by_res_layers)
        for res_idx, (res_vars, rnn_part) in enumerate(
                zip(self._opt_trainable['res_layers'], rnn_output_by_res_layers)):
            optimizer_ins, rnn_input_part = self._apply_res_layer(
                optimizer_ins, res_vars, rnn_part, 'res_layer_%s' % res_idx)
            rnn_input_by_res_layers.append(rnn_input_part)

        optimizer_ins = self._split_opt_ins(optimizer_ins, ['o_c', 'sigma_c'], num_concatenated)
        optimizer_outs = self._mv_tensors(optimizer_ins, ['o_c_spl', 'sigma_c_spl'], ['o_pr', 'sigma_pr'])
        optimizer_outs = self._backward_permute(optimizer_outs, ['o_pr'], ['sigma_pr'])

        rnn_input = tf.concat(rnn_input_by_res_layers, -1)
        rnn_input = tf.reduce_mean(rnn_input, axis=-2)
        state = self._apply_lstm_layer(
            rnn_input, state, self._opt_trainable['lstm_matrix'], self._opt_trainable['lstm_bias'])
        return optimizer_outs, state

    def __init__(self,
                 pupil,
                 num_exercises=10,
                 num_lstm_nodes=256,
                 num_optimizer_unrollings=10,
                 perm_period=None,
                 num_res_layers=4,
                 res_size=1000,
                 num_gpus=1,
                 regime='train',
                 optimizer_for_opt_type='adam'):
        self._pupil = pupil
        self._pupil_net_size = self._pupil.get_net_size()
        self._pupil_dims = self._pupil.get_layer_dims()
        self._emb_layer_is_present = 'embedding_size' in self._pupil_net_size
        self._num_exercises = num_exercises
        self._num_lstm_nodes = num_lstm_nodes
        self._num_optimizer_unrollings = num_optimizer_unrollings
        self._perm_period = perm_period
        self._num_res_layers = num_res_layers
        self._res_size = res_size
        self._num_gpus = num_gpus
        if self._num_gpus == 1:
            self._base_device = '/gpu:0'
        else:
            self._base_device = '/cpu:0'
        self._regime = regime

        self._optimizer_for_opt_type = optimizer_for_opt_type

        self._hooks = dict(
            pupil_grad_eval_inputs=None,
            pupil_grad_eval_labels=None,
            optimizer_grad_inputs=None,
            optimizer_grad_labels=None,
            pupil_savers=None,
            optimizer_train_op=None,
            learning_rate_for_optimizer_training=None,
            train_with_meta_optimizer_op=None,
            reset_train_states=None,
            reset_inference_state=None,
            reset_permutation_matrices=None,
            reset_pupil_grad_eval_pupil_storage=None,
            reset_optimizer_grad_pupil_storage=None,
            meta_optimizer_saver=None,
            loss=None,
            start_loss=None,
            end_loss=None,
            optimizer_dropout_keep_prob=None,
            pupil_trainable_initializers=None
        )

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
            self._hooks['pupil_trainable_initializers'] = self._pupil_trainable_initializers
            self._hooks['reset_pupil_grad_eval_pupil_storage'] = tf.group(
                *self._pupil.reset_storage(
                    self._pupil_grad_eval_pupil_storage)
            )
            self._hooks['reset_optimizer_grad_pupil_storage'] = tf.group(
                *self._pupil.reset_storage(
                    self._optimizer_grad_pupil_storage)
            )
            self._add_standard_train_hooks()
        else:
            self._exercise_gpu_map = None
            self._pupil_grad_eval_inputs, self._pupil_grad_eval_labels, \
                self._optimizer_grad_inputs, self._optimizer_grad_labels = None, None, None, None

        self._rnn_for_res_layers = self._distribute_rnn_size_among_res_layers(
            self._num_lstm_nodes, self._num_res_layers)
        with tf.device(self._base_device):
            self._opt_trainable = self._create_optimizer_trainable_vars()

        self._create_permutation_matrices(1, 0)

        if self._regime == 'train':
            self._learning_rate_for_optimizer_training = tf.placeholder(
                tf.float32, name='learning_rate_for_optimizer_training')
            if self._optimizer_for_opt_type == 'adam':
                self._optimizer_for_optimizer_training = tf.train.AdamOptimizer(
                    learning_rate=self._learning_rate_for_optimizer_training)
            elif self._optimizer_for_opt_type == 'sgd':
                self._optimizer_for_optimizer_training = tf.train.GradientDescentOptimizer(
                    learning_rate=self._learning_rate_for_optimizer_training)
            self._train_graph()
            self._inference_graph()
            self._hooks['reset_train_states'] = tf.group(
                *[self._reset_optimizer_states('train', gpu_idx)
                  for gpu_idx in range(self._num_gpus)])
            self._hooks['reset_inference_state'] = self._reset_optimizer_states('inference', 0)

            self._hooks['reset_permutation_matrices'] = tf.group(*self._reset_all_permutation_matrices())
        elif self._regime == 'inference':
            self._inference_graph()
            self._hooks['reset_inference_state'] = self._reset_optimizer_states('inference', 0)

    def get_default_hooks(self):
        return construct_dict_without_none_entries(self._hooks)
