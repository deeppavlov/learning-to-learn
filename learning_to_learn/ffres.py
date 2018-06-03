from itertools import chain

import tensorflow as tf
from learning_to_learn.useful_functions import block_diagonal, custom_matmul, custom_add, flatten, \
    construct_dict_without_none_entries, construct

from learning_to_learn.meta import Meta


class FfResOpt(Meta):

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

    @staticmethod
    def _distribute_rnn_size_among_res_layers(rnn_size, num_res_layers):
        rnn_nodes_for_layer = rnn_size // num_res_layers
        remainder = rnn_size % num_res_layers
        return [rnn_nodes_for_layer for _ in range(num_res_layers-1)] + [rnn_nodes_for_layer + remainder]

    def _res_core_vars(self, target, res_size, var_scope):
        with tf.variable_scope(var_scope):
            matrices, biases = list(), list()
            in_ndims = sum(flatten(target))
            out_ndims = sum(flatten(target))
            in_stddev = self._optimizer_init_parameter / (in_ndims + res_size)**.5
            with tf.variable_scope('in_core'):
                matrices.append(
                    tf.get_variable(
                        'matrix',
                        shape=[in_ndims, res_size],
                        initializer=tf.truncated_normal_initializer(stddev=in_stddev),
                        # initializer=tf.zeros_initializer(),
                        # trainable=False
                    )
                )
                biases.append(
                    tf.get_variable(
                        'bias',
                        shape=[res_size],
                        initializer=tf.zeros_initializer(),
                        # trainable=False
                    )
                )
            with tf.variable_scope('out_core'):
                matrices.append(
                    tf.get_variable(
                        'matrix',
                        # shape=[res_size, out_ndims],
                        # initializer=tf.truncated_normal_initializer(stddev=in_stddev)
                        initializer=tf.zeros_initializer()
                        # trainable=False
                    )
                )
                biases.append(
                    tf.get_variable(
                        'bias',
                        shape=[out_ndims],
                        # initializer=tf.zeros_initializer(),
                        initializer=tf.constant_initializer(1e-15),  # because otherwise neurons will be dead
                        # initializer=tf.truncated_normal_initializer(stddev=in_stddev)
                        # trainable=False
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
        vars = dict()
        with tf.variable_scope('optimizer_trainable_variables'):
            with tf.variable_scope('res_layers'):
                vars['res_layers'] = list()
                for res_idx in range(self._num_res_layers):
                    with tf.variable_scope('layer_%s' % res_idx):
                        res_layer_params = dict()
                        if self._emb_layer_is_present:
                            res_layer_params['embedding_layer'] = self._res_core_vars(
                                [embedding_layer],
                                self._res_size,
                                'embedding_layer_core'
                            )
                        for layer_idx, layer_dims in enumerate(lstm_layers):
                            res_layer_params['lstm_layer_%s' % layer_idx] = self._res_core_vars(
                                [layer_dims],
                                self._res_size,
                                'lstm_layer_%s_core' % layer_idx
                            )
                        for layer_idx, layer_dims in enumerate(output_layers):
                            res_layer_params['output_layer_%s' % layer_idx] = self._res_core_vars(
                                [layer_dims],
                                self._res_size,
                                'output_layer_%s_core' % layer_idx
                            )
                        vars['res_layers'].append(res_layer_params)
        return vars

    # def _optimizer_core(self, optimizer_ins, num_exercises, states, gpu_idx):
    #     # optimizer_ins = self._extend_with_permutations(optimizer_ins, num_exercises, gpu_idx)
    #     # optimizer_ins = self._forward_permute(optimizer_ins)
    #     return self._empty_core(optimizer_ins)


    # @staticmethod
    # def _apply_res_core(vars, opt_ins, rnn_part, target, scope, target_dims):
    def _apply_res_core(self, vars, opt_ins, scope, target_dims):
        if self._res_core_activation_func == 'relu':
            a_func = tf.nn.relu
        elif self._res_core_activation_func == 'tanh':
            a_func = tf.tanh
        else:
            a_func = None
        with tf.name_scope(scope):
            # print("\n(ResNet4Lstm._apply_res_core)rnn_part:", rnn_part)
            # print('(ResNet4Lstm._apply_res_core)opt_ins:', opt_ins)
            opt_ins_united = tf.concat(opt_ins, -1, name='opt_ins_united')
            # print("(ResNet4Lstm._apply_res_core)rnn_part:", rnn_part)
            hs = opt_ins_united
            matrices = vars[0]
            biases = vars[1]
            for idx, (m, b) in enumerate(zip(matrices, biases)):
                # print('\n(ResNet4Lstm._apply_res_core)hs:', hs)
                # print('(ResNet4Lstm._apply_res_core)m:', m)
                # with tf.device('/cpu:0'):
                #     hs = tf.Print(
                #         hs, [l2_loss_per_elem(hs)],
                #         message="(ResNetOpt._apply_res_core)(%s)(%s)hs before: " % (scope, idx), summarize=20)

                # if scope == 'lstm_layer_0':
                #     with tf.device('/cpu:0'):
                #         m = tf.Print(
                #             m, [tf.sqrt(l2_loss_per_elem(m))],
                #             message="(ResNet4Lstm._apply_res_core)matrix_%s_%s: " % (scope, idx)
                #         )
                #         hs = tf.Print(
                #             hs, [tf.sqrt(l2_loss_per_elem(hs))],
                #             message="(ResNet4Lstm._apply_res_core)hs_%s_%s: " % (scope, idx)
                #         )

                matmul_res = custom_matmul(hs, m)
                if idx == 0:
                    self._debug_tensors.append(matmul_res)
                hs = a_func(custom_add(matmul_res, b))
                # hs = tf.tanh(custom_add(matmul_res, b))

                # with tf.device('/cpu:0'):
                #     hs = tf.Print(
                #         hs, [l2_loss_per_elem(matmul_res)],
                #         message="(ResNetOpt._apply_res_core)(%s)(%s)matmul_res: " % (scope, idx), summarize=20)
            hs = tf.add(
                hs,
                tf.concat(opt_ins, -1, name='res_tensor'),
                name='after_res_conn'
            )
            o, sigma = tf.split(hs, list(target_dims), axis=-1, name='o_sigma')
            # print("(ResNet4Lstm._apply_res_core)rnn_part:", rnn_part)
            # with tf.device('/cpu:0'):
            #     o = tf.Print(o, [o], message='(ResNetOpt._apply_res_core)(scope=%s)o: ' % scope, summarize=10)
            #     sigma = tf.Print(
            #         sigma, [sigma], message='(ResNetOpt._apply_res_core)(scope=%s)sigma: ' % scope, summarize=10)
            return o, sigma

    def _apply_res_layer(self, ins, res_vars, scope):
        with tf.name_scope(scope):
            outs = construct(ins)
            if self._emb_layer_is_present:
                target = [
                    ins['embedding_layer']['o_c'],
                    ins['embedding_layer']['sigma_c']
                ]
                # print('(ResNet4Lstm._apply_res_layer)core_inps:', core_inps)
                # print('(ResNet4Lstm._apply_res_layer)res_vars:', res_vars)
                o, sigma = self._apply_res_core(
                    res_vars['embedding_layer'], target,
                    'embedding_layer', self._pupil_dims['embedding_layer'])
                outs['embedding_layer']['o_c'] = o
                outs['embedding_layer']['sigma_c'] = sigma

            for layer_idx in range(self._pupil_net_size['num_layers']):
                layer_name = 'lstm_layer_%s' % layer_idx
                target = [
                    ins[layer_name]['o_c'],
                    ins[layer_name]['sigma_c']
                ]
                o, sigma = self._apply_res_core(
                    res_vars[layer_name], target,
                    layer_name, self._pupil_dims['lstm_layers'][layer_idx])
                outs[layer_name]['o_c'] = o
                outs[layer_name]['sigma_c'] = sigma

            for layer_idx in range(self._pupil_net_size['num_output_layers']):
                target = [
                    ins['output_layer_%s' % layer_idx]['o_c'],
                    ins['output_layer_%s' % layer_idx]['sigma_c']
                ]
                layer_name = 'output_layer_%s' % layer_idx
                # print("(ResNet4Lstm._apply_res_layer)layer_idx:", layer_idx)
                # print("(ResNet4Lstm._apply_res_layer)core_inps:", core_inps)
                # print("(ResNet4Lstm._apply_res_layer)rnn_part:", rnn_part)
                o, sigma = self._apply_res_core(
                    res_vars[layer_name], target,
                    layer_name, self._pupil_dims['output_layers'][layer_idx])
                layer_name = 'output_layer_%s' % layer_idx
                outs[layer_name]['o_c'] = o
                outs[layer_name]['sigma_c'] = sigma
            # print("(ResNet4Lstm._apply_res_layer)emb_rnn_part:", emb_rnn_part)
            # print("(ResNet4Lstm._apply_res_layer)lstm_rnn_parts:", lstm_rnn_parts)
            # print("(ResNet4Lstm._apply_res_layer)output_rnn_parts:", output_rnn_parts)
            return outs

    def _optimizer_core(self, optimizer_ins, state, gpu_idx, permute=True):
        if permute:
            optimizer_ins = self._extend_with_permutations(optimizer_ins, gpu_idx)
        # print('(ResNet4Lstm._optimizer_core)optimizer_ins\nBEFORE DIMS EXPANSION:')
        # print_optimizer_ins(optimizer_ins)
        # ndims = self._get_optimizer_ins_ndims(optimizer_ins)
        # if ndims == 2:
        #     optimizer_ins = self._expand_num_ex_dim_in_opt_ins(optimizer_ins, ['o', 'sigma'])

        # print('(ResNet4Lstm._optimizer_core)optimizer_ins\nBEFORE PERMUTATION:')
        # print_optimizer_ins(optimizer_ins)
        if permute:
            optimizer_ins = self._forward_permute(optimizer_ins, ['o'], ['sigma'])
        # print('(ResNet4Lstm._optimizer_core)optimizer_ins\nBEFORE CONCATENATION:')
        # print_optimizer_ins(optimizer_ins)
        optimizer_ins, num_concatenated = self._concat_opt_ins(optimizer_ins, ['o', 'sigma'])
        # print('(ResNet4Lstm._optimizer_core)optimizer_ins\nAFTER CONCATENATION:')
        # print_optimizer_ins(optimizer_ins)
        # print("(ResNet4Lstm._optimizer_core)self._opt_trainable['res_layers']:", self._opt_trainable['res_layers'])
        # print('(ResNet4Lstm._optimizer_core)rnn_output_by_res_layers:', rnn_output_by_res_layers)
        for res_idx, res_vars in enumerate(self._opt_trainable['res_layers']):
            optimizer_ins = self._apply_res_layer(
                optimizer_ins, res_vars, 'res_layer_%s' % res_idx)

        optimizer_ins = self._split_opt_ins(optimizer_ins, ['o_c', 'sigma_c'], num_concatenated)
        optimizer_outs = self._mv_tensors(optimizer_ins, ['o_c_spl', 'sigma_c_spl'], ['o_pr', 'sigma_pr'])
        if permute:
            optimizer_outs = self._backward_permute(optimizer_outs, ['o_pr'], ['sigma_pr'])
        return optimizer_outs, state

    def __init__(
            self,
            pupil,
            num_exercises=10,
            num_optimizer_unrollings=10,
            perm_period=None,
            num_res_layers=4,
            res_size=1000,
            res_core_activation_func='relu',
            num_gpus=1,
            regularization_rate=6e-6,
            clip_norm=1e+5,
            optimizer_init_parameter=.1,
            permute=True,
            regime='train',
            optimizer_for_opt_type='adam',
            additional_metrics=None,
            opt_ins_substitution=None
    ):
        if additional_metrics is None:
            additional_metrics = list()

        self._pupil = pupil
        self._pupil_net_size = self._pupil.get_net_size()
        self._pupil_dims = self._pupil.get_layer_dims()
        self._emb_layer_is_present = 'embedding_size' in self._pupil_net_size
        self._num_exercises = num_exercises
        self._num_optimizer_unrollings = num_optimizer_unrollings
        self._perm_period = perm_period
        self._num_res_layers = num_res_layers
        self._res_size = res_size
        self._res_core_activation_func = res_core_activation_func
        self._num_gpus = num_gpus
        if self._num_gpus == 1:
            self._base_device = '/gpu:0'
        else:
            self._base_device = '/cpu:0'
        self._regularization_rate = regularization_rate
        self._clip_norm = clip_norm
        self._optimizer_init_parameter = optimizer_init_parameter
        self._permute = permute
        self._regime = regime

        self._optimizer_for_opt_type = optimizer_for_opt_type

        self._additional_metrics = additional_metrics

        self._opt_ins_substitution = opt_ins_substitution

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
            pupil_trainable_initializers=None
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
            # print("(ResNet4Lstm.__init__)self._pupil_grad_eval_pupil_storage:", self._pupil_grad_eval_pupil_storage)
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

        # self._create_permutation_matrices(1, 0)

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
            # print([self._reset_optimizer_states('train', gpu_idx) for gpu_idx in range(self._num_gpus)])
            self._hooks['reset_optimizer_train_state'] = tf.group(
                *chain(
                    *[self._reset_optimizer_states('train_optimizer_states', gpu_idx)
                      for gpu_idx in range(self._num_gpus)]))
            self._hooks['reset_optimizer_inference_state'] = self._reset_optimizer_states(
                'inference_optimizer_states', 0)[0]

            self._hooks['reset_permutation_matrices'] = tf.group(
                *self._reset_all_permutation_matrices())
        elif self._regime == 'inference':
            self._inference_graph()
            self._hooks['reset_optimizer_inference_state'] = self._reset_optimizer_states(
                'inference_optimizer_states', 0)

    def get_default_hooks(self):
        return construct_dict_without_none_entries(self._hooks)
