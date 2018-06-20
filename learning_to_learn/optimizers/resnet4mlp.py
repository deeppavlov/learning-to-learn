from itertools import chain

import tensorflow as tf

from learning_to_learn.optimizers.meta import Meta
from learning_to_learn.useful_functions import block_diagonal, custom_matmul, custom_add, flatten, \
    construct_dict_without_none_entries, construct


class ResNet4Mlp(Meta):

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
                # print("(ResNet4Mlp._reset_optimizer_states)h:", h)
                # print("(ResNet4Mlp._reset_optimizer_states)c:", c)
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
        with tf.variable_scope('permutation_matrices_on_gpu_%s' % gpu_idx):
            for layer_idx in range(self._pupil_net_size['num_layers'] - 1):
                _ = tf.get_variable(
                    'h_%s' % layer_idx,
                    initializer=self._create_permutation_matrix(num_nodes[layer_idx], num_exercises),
                    trainable=False
                )

    def _reset_all_permutation_matrices(self):
        reset_ops = list()
        for gpu_idx, num_exercises in enumerate(self._num_ex_on_gpus):
            reset_ops.extend(self._reset_permutations(gpu_idx))
        return reset_ops

    @staticmethod
    def _distribute_rnn_size_among_res_layers(rnn_size, num_res_layers):
        rnn_nodes_for_layer = rnn_size // num_res_layers
        remainder = rnn_size % num_res_layers
        return [rnn_nodes_for_layer for _ in range(num_res_layers-1)] + [rnn_nodes_for_layer + remainder]

    def _res_core_vars(self, left, target, right, rnn_part_size, res_size, var_scope):
        with tf.variable_scope(var_scope):
            matrices, biases = list(), list()
            in_ndims = sum(flatten(left)) + sum(flatten(target)) + sum(flatten(right)) + rnn_part_size
            out_ndims = sum(flatten(target)) + rnn_part_size
            out_stddev = self._optimizer_init_parameter / (res_size + rnn_part_size)**.5
            out_init = tf.concat(
                [tf.zeros([res_size, sum(flatten(target))]),
                 tf.truncated_normal([res_size, rnn_part_size], stddev=out_stddev)],
                -1
            )
            in_stddev = self._optimizer_init_parameter / (in_ndims + res_size)**.5
            with tf.variable_scope('in_core'):
                matrices.append(
                    0*tf.get_variable(
                        'matrix',
                        shape=[in_ndims, res_size],
                        initializer=tf.truncated_normal_initializer(stddev=in_stddev),
                        # initializer=tf.zeros_initializer(),
                        # trainable=False
                    )
                )
                biases.append(
                    0*tf.get_variable(
                        'bias',
                        shape=[res_size],
                        initializer=tf.zeros_initializer(),
                        # trainable=False
                    )
                )
            with tf.variable_scope('out_core'):
                matrices.append(
                0*tf.get_variable(
                        'matrix',
                        # shape=[res_size, out_ndims],
                        # initializer=tf.truncated_normal_initializer(stddev=in_stddev)
                        initializer=out_init,
                        # initializer=tf.zeros_initializer()
                        # trainable=False
                    )
                )
                biases.append(
                    0*tf.get_variable(
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
        layers = self._pupil_dims['layers']
        num_layers = self._pupil_net_size['num_layers']
        vars = dict()
        with tf.variable_scope('optimizer_trainable_variables'):
            with tf.variable_scope('res_layers'):
                vars['res_layers'] = list()
                for res_idx, rnn_part in enumerate(self._rnn_for_res_layers):
                    with tf.variable_scope('layer_%s' % res_idx):
                        res_layer_params = dict()
                        for layer_idx, layer_dims in enumerate(layers):
                            if layer_idx == 0:
                                pupil_previous_layer_dims = ()
                            else:
                                pupil_previous_layer_dims = layers[layer_idx-1]
                            if layer_idx == num_layers - 1:
                                pupil_next_layer_dims = ()
                            else:
                                pupil_next_layer_dims = layers[layer_idx+1]
                            res_layer_params['layer_%s' % layer_idx] = self._res_core_vars(
                                [pupil_previous_layer_dims],
                                [layer_dims],
                                [pupil_next_layer_dims],
                                rnn_part,
                                self._res_size,
                                'layer_%s_core' % layer_idx
                            )
                        vars['res_layers'].append(res_layer_params)
                with tf.variable_scope('lstm'):
                    rnn_size = self._num_lstm_nodes
                    stddev = 1. / (6 * rnn_size) ** .5
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

    # @staticmethod
    # def _apply_res_core(vars, opt_ins, rnn_part, target, scope, target_dims):
    def _apply_res_core(self, vars, opt_ins, rnn_part, target, scope, target_dims):
        if self._res_core_activation_func == 'relu':
            a_func = tf.nn.relu
        elif self._res_core_activation_func == 'tanh':
            a_func = tf.tanh
        else:
            a_func = None
        with tf.name_scope(scope):
            # print("\n(ResNet4Mlp._apply_res_core)rnn_part:", rnn_part)
            # print('(ResNet4Mlp._apply_res_core)opt_ins_united:', opt_ins_united)
            opt_ins_united = tf.concat(opt_ins, -1, name='opt_ins_united')
            ndim = len(opt_ins_united.get_shape().as_list())
            # print('(ResNet4Mlp._apply_res_core)opt_ins:', opt_ins)
            rnn_stack_num = tf.concat(
                [tf.ones([1], dtype=tf.int32),
                 tf.shape(opt_ins_united)[-2:-1],
                 tf.ones([1], dtype=tf.int32)],
                0
            )
            rnn_part = tf.expand_dims(rnn_part, 1)
            rnn_part = tf.tile(rnn_part, rnn_stack_num, name='stacked_rnn_part')
            # print("(ResNet4Mlp._apply_res_core)rnn_part:", rnn_part)
            hs = tf.concat([opt_ins_united, rnn_part], -1, name='opt_ins_with_rnn_part')
            matrices = vars[0]
            biases = vars[1]
            for idx, (m, b) in enumerate(zip(matrices, biases)):
                # print('\n(ResNet4Mlp._apply_res_core)hs:', hs)
                # print('(ResNet4Mlp._apply_res_core)m:', m)
                # with tf.device('/cpu:0'):
                #     hs = tf.Print(
                #         hs, [l2_loss_per_elem(hs)],
                #         message="(ResNetOpt._apply_res_core)(%s)(%s)hs before: " % (scope, idx), summarize=20)


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
                tf.concat(target + [tf.zeros(tf.shape(rnn_part))], -1, name='res_tensor'),
                name='after_res_conn'
            )
            rnn_part_dim = hs.get_shape().as_list()[-1] - sum(target_dims)
            o, sigma, rnn_part = tf.split(hs, list(target_dims) + [rnn_part_dim], axis=-1, name='o_sigma_and_rnn_part')
            # print("(ResNet4Mlp._apply_res_core)rnn_part:", rnn_part)
            # with tf.device('/cpu:0'):
            #     o = tf.Print(o, [o], message='(ResNetOpt._apply_res_core)(scope=%s)o: ' % scope, summarize=10)
            #     sigma = tf.Print(
            #         sigma, [sigma], message='(ResNetOpt._apply_res_core)(scope=%s)sigma: ' % scope, summarize=10)
            return o, sigma, rnn_part

    def _apply_res_layer(self, ins, res_vars, rnn_part, scope):
        with tf.name_scope(scope):
            outs = construct(ins)
            rnn_parts = list()
            for layer_idx in range(self._pupil_net_size['num_layers']):
                if layer_idx == 0:
                    previous_layer_tensors = []
                else:
                    previous_layer_tensors = [
                        ins['layer_%s' % (layer_idx - 1)]['o'],
                        ins['layer_%s' % (layer_idx - 1)]['sigma']
                    ]
                if layer_idx == self._pupil_net_size['num_layers'] - 1:
                    next_layer_tensors = []
                else:
                    next_layer_tensors = [
                        ins['layer_%s' % (layer_idx + 1)]['o'],
                        ins['layer_%s' % (layer_idx + 1)]['sigma']
                    ]
                layer_name = 'layer_%s' % layer_idx
                core_inps = [
                    *previous_layer_tensors,
                    ins[layer_name]['o'],
                    ins[layer_name]['sigma'],
                    *next_layer_tensors
                ]
                target = [
                    ins[layer_name]['o'],
                    ins[layer_name]['sigma']
                ]
                o, sigma, lstm_rnn_part = self._apply_res_core(
                    res_vars[layer_name], core_inps, rnn_part, target,
                    layer_name, self._pupil_dims['layers'][layer_idx])
                rnn_parts.append(lstm_rnn_part)
                outs[layer_name]['o'] = o
                outs[layer_name]['sigma'] = sigma

            # print("(ResNet4Lstm._apply_res_layer)emb_rnn_part:", emb_rnn_part)
            # print("(ResNet4Lstm._apply_res_layer)lstm_rnn_parts:", lstm_rnn_parts)
            # print("(ResNet4Lstm._apply_res_layer)output_rnn_parts:", output_rnn_parts)
            return outs, sum(rnn_parts)

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

    def _optimizer_core(self, optimizer_ins, state, gpu_idx, permute=True):
        # print('(ResNet4Mlp._optimizer_core)optimizer_ins\nBEFORE DIMS EXPANSION:')
        # print_optimizer_ins(optimizer_ins)
        # ndims = self._get_optimizer_ins_ndims(optimizer_ins)
        # if ndims == 2:
        #     optimizer_ins = self._expand_num_ex_dim_in_opt_ins(optimizer_ins, ['o', 'sigma'])

        # print('(ResNet4Mlp._optimizer_core)optimizer_ins\nBEFORE PERMUTATION:')
        # print_optimizer_ins(optimizer_ins)
        # with tf.device('/cpu:0'):
        #     for ok, ov in optimizer_ins.items():
        #         for ik, iv in ov.items():
        #             if ik in ['o', 'sigma']:
        #                 msg = '\n\nins\n' + ' ' * 4 + ok + ':\n' + ' ' * 2 + ik + ':\n'
        #                 if isinstance(iv, list):
        #                     for idx, v in enumerate(iv):
        #                         iv[idx] = tf.Print(
        #                             v,
        #                             [v],
        #                             message=msg + '%s' % idx + '\n',
        #                             summarize=20,
        #                         )
        #                 else:
        #                     ov[ik] = tf.Print(
        #                         iv,
        #                         [iv],
        #                         message=msg,
        #                         summarize=20,
        #                     )

        if permute:
            pass
            # optimizer_ins = self._extend_with_permutations(optimizer_ins, gpu_idx)
            # optimizer_ins = self._forward_permute(optimizer_ins, ['o'], ['sigma'])
        # print('(ResNet4Mlp._optimizer_core)optimizer_ins\nBEFORE CONCATENATION:')
        # print_optimizer_ins(optimizer_ins)
        # print('(ResNet4Mlp._optimizer_core)optimizer_ins\nAFTER CONCATENATION:')
        # print_optimizer_ins(optimizer_ins)
        rnn_output_by_res_layers = tf.split(state[0], self._rnn_for_res_layers, axis=-1)
        # print("(ResNet4Mlp._optimizer_core)self._opt_trainable['res_layers']:", self._opt_trainable['res_layers'])
        rnn_input_by_res_layers = list()
        # print('(ResNet4Mlp._optimizer_core)rnn_output_by_res_layers:', rnn_output_by_res_layers)
        for res_idx, (res_vars, rnn_part) in enumerate(
                zip(self._opt_trainable['res_layers'], rnn_output_by_res_layers)):
            optimizer_ins, rnn_input_part = self._apply_res_layer(
                optimizer_ins, res_vars, rnn_part, 'res_layer_%s' % res_idx)
            rnn_input_by_res_layers.append(rnn_input_part)

        optimizer_outs = self._mv_tensors(optimizer_ins, ['o', 'sigma'], ['o_pr', 'sigma_pr'])
        if permute:
            pass
            # optimizer_outs = self._backward_permute(optimizer_outs, ['o_pr'], ['sigma_pr'])
        # with tf.device('/cpu:0'):
        #     for ok, ov in optimizer_ins.items():
        #         for ik, iv in ov.items():
        #             if ik in ['o_pr', 'sigma_pr']:
        #                 msg = '\n\nouts\n' + ' ' * 4 + ok + ':\n' + ' ' * 2 + ik + ':\n'
        #                 if isinstance(iv, list):
        #                     for idx, v in enumerate(iv):
        #                         iv[idx] = tf.Print(
        #                             v,
        #                             [v],
        #                             message=msg + '%s' % idx + '\n',
        #                             summarize=20,
        #                         )
        #                 else:
        #                     ov[ik] = tf.Print(
        #                         iv,
        #                         [iv],
        #                         message=msg,
        #                         summarize=20,
        #                     )
        rnn_input = tf.concat(rnn_input_by_res_layers, -1, name='all_for_rnn')
        rnn_input = tf.reduce_mean(rnn_input, axis=-2, name='rnn_input')
        state = self._apply_lstm_layer(
            rnn_input, state, self._opt_trainable['lstm_matrix'], self._opt_trainable['lstm_bias'])
        self._multiply_by_factor(
            optimizer_outs,
            dict(
                sigma_pr=self._pupil_learning_rate
            )
        )
        return optimizer_outs, state

    def __init__(
            self,
            pupil,
            num_exercises=10,
            num_lstm_nodes=256,
            num_optimizer_unrollings=10,
            perm_period=None,
            num_res_layers=4,
            res_size=1000,
            res_core_activation_func='relu',
            num_gpus=1,
            regularization_rate=6e-6,
            pupil_learning_rate=1e-4,
            inp_gradient_clipping='norm_loss',
            clip_norm=1e+5,
            optimizer_init_parameter=.1,
            permute=True,
            regime='train',
            optimizer_for_opt_type='adam',
            additional_metrics=None,
            flags=None,
            normalizing=None,
            get_theta=False,
            get_omega_and_beta=False,
            matrix_mod='phi_and_psi',
    ):
        """
        :param pupil:
        :param num_exercises:
        :param num_lstm_nodes:
        :param num_optimizer_unrollings:
        :param perm_period:
        :param num_res_layers:
        :param res_size:
        :param res_core_activation_func:
        :param num_gpus:
        :param regularization_rate:
        :param clip_norm:
        :param optimizer_init_parameter:
        :param permute:
        :param regime:
        :param optimizer_for_opt_type:
        :param additional_metrics:
        :param flags: a list containing some of the following
            'summarize_opt_ins': if present summary operations for optimizer inputs ('o', 's', 'sigma') are created
            'opt_ins_substitution': if present optimizer ins will be replaced with constant tensors. To specify them
                got to line "if 'opt_ins_substitution' in self._flags:" and choose your option
        :param normalizing: a dictionary. Specifies the way optimizer ins are normalized.
            'type': normalizing type.
            several other entries.
            Now only 'factor' option is present.
            If specified element of optimizer ins ('sigma', 'o' etc.) is multiplied by corresponding element of
            normalizing['factors'] dictionary.
        """
        if additional_metrics is None:
            additional_metrics = list()
        if flags is None:
            flags = list()

        self._pupil = pupil
        self._pupil_net_size = self._pupil.get_net_size()
        self._pupil_dims = self._pupil.get_layer_dims()
        self._num_exercises = num_exercises
        self._num_lstm_nodes = num_lstm_nodes
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
        self._inp_gradient_clipping = inp_gradient_clipping
        self._clip_norm = clip_norm
        self._optimizer_init_parameter = optimizer_init_parameter
        self._permute = permute
        self._regime = regime

        self._optimizer_for_opt_type = optimizer_for_opt_type
        self._pupil_learning_rate = pupil_learning_rate
        self._additional_metrics = additional_metrics

        self._flags = flags
        self._get_theta = get_theta
        self._get_omega_and_beta = get_omega_and_beta
        self._matrix_mod = matrix_mod

        self._normalizing = normalizing

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
            # print("(ResNet4Mlp.__init__)self._pupil_grad_eval_pupil_storage:", self._pupil_grad_eval_pupil_storage)
            self._hooks['pupil_savers'] = self._pupil_savers
            self._hooks['pupil_trainable_initializers'] = self._pupil_trainable_initializers
            self._hooks['reset_pupil_grad_eval_pupil_storage'] = tf.group(
                *chain(*[self._pupil.reset_storage(stor) for stor in self._pupil_grad_eval_pupil_storage])
            )
            self._hooks['reset_optimizer_grad_pupil_storage'] = tf.group(
                *chain(*[self._pupil.reset_storage(stor) for stor in self._optimizer_grad_pupil_storage])
            )
            self._add_standard_train_hooks()

            self._additional_loss = 0
        else:
            self._exercise_gpu_map = None
            self._pupil_grad_eval_inputs, self._pupil_grad_eval_labels, \
                self._optimizer_grad_inputs, self._optimizer_grad_labels = None, None, None, None

        self._rnn_for_res_layers = self._distribute_rnn_size_among_res_layers(
            self._num_lstm_nodes, self._num_res_layers)
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
