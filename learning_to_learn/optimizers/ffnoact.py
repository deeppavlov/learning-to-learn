from itertools import chain

import tensorflow as tf

from learning_to_learn.optimizers.meta import Meta
from learning_to_learn.useful_functions import block_diagonal, custom_matmul, custom_add, flatten, \
    construct_dict_without_none_entries, construct


class Ff(Meta):

    @staticmethod
    def check_kwargs(**kwargs):
        pass

    @staticmethod
    def _create_optimizer_states(*args):
        return list()

    @staticmethod
    def _reset_optimizer_states(var_scope, gpu_idx):
        return list()

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

    def _core_vars(self, target, var_scope):
        with tf.variable_scope(var_scope):
            ndims = sum(flatten(target))
            stddev = self._optimizer_init_parameter / (2 * ndims)**.5
            matrix = tf.get_variable(
                'matrix',
                shape=[ndims, ndims],
                initializer=tf.truncated_normal_initializer(stddev=stddev),
            )
            bias = tf.get_variable(
                'bias',
                shape=[ndims],
                initializer=tf.zeros_initializer(),
                # trainable=False
            )
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, matrix)
        return matrix, bias

    def _create_optimizer_trainable_vars(self):
        if self._emb_layer_is_present:
            embedding_layer = self._pupil_dims['embedding_layer']
        lstm_layers = self._pupil_dims['lstm_layers']
        output_layers = self._pupil_dims['output_layers']
        vars = list()
        with tf.variable_scope('optimizer_trainable_variables'):
            for layer_idx in range(self._num_layers):
                with tf.variable_scope('layer_%s' % layer_idx):
                    layer_params = dict()
                    if self._emb_layer_is_present:
                        layer_params['embedding_layer'] = self._core_vars(
                            [embedding_layer],
                            'embedding_layer_core'
                        )
                    for layer_idx, layer_dims in enumerate(lstm_layers):
                        layer_params['lstm_layer_%s' % layer_idx] = self._core_vars(
                            [layer_dims],
                            'lstm_layer_%s_core' % layer_idx
                        )
                    for layer_idx, layer_dims in enumerate(output_layers):
                        layer_params['output_layer_%s' % layer_idx] = self._core_vars(
                            [layer_dims],
                            'output_layer_%s_core' % layer_idx
                        )
                    vars.append(layer_params)
        return vars

    # def _optimizer_core(self, optimizer_ins, num_exercises, states, gpu_idx):
    #     # optimizer_ins = self._extend_with_permutations(optimizer_ins, num_exercises, gpu_idx)
    #     # optimizer_ins = self._forward_permute(optimizer_ins)
    #     return self._empty_core(optimizer_ins)

    def _apply_core(self, vars, opt_ins, scope, target_dims):
        with tf.name_scope(scope):
            opt_ins_united = tf.concat(opt_ins, -1, name='opt_ins_united')
            hs = opt_ins_united
            matrix = vars[0]
            bias = vars[1]
            matmul_res = custom_matmul(hs, matrix)
            self._debug_tensors.append(matmul_res)
            hs = custom_add(matmul_res, bias)
            o, sigma = tf.split(hs, list(target_dims), axis=-1, name='o_sigma')
            return o, sigma

    def _apply_layer(self, ins, vars, scope):
        with tf.name_scope(scope):
            outs = construct(ins)
            if self._emb_layer_is_present:
                target = [
                    ins['embedding_layer']['o_c'],
                    ins['embedding_layer']['sigma_c']
                ]
                o, sigma = self._apply_core(
                    vars['embedding_layer'], target,
                    'embedding_layer', self._pupil_dims['embedding_layer'])
                outs['embedding_layer']['o_c'] = o
                outs['embedding_layer']['sigma_c'] = sigma

            for layer_idx in range(self._pupil_net_size['num_layers']):
                layer_name = 'lstm_layer_%s' % layer_idx
                target = [
                    ins[layer_name]['o_c'],
                    ins[layer_name]['sigma_c']
                ]
                o, sigma = self._apply_core(
                    vars[layer_name], target,
                    layer_name, self._pupil_dims['lstm_layers'][layer_idx])
                outs[layer_name]['o_c'] = o
                outs[layer_name]['sigma_c'] = sigma

            for layer_idx in range(self._pupil_net_size['num_output_layers']):
                target = [
                    ins['output_layer_%s' % layer_idx]['o_c'],
                    ins['output_layer_%s' % layer_idx]['sigma_c']
                ]
                layer_name = 'output_layer_%s' % layer_idx
                o, sigma = self._apply_core(
                    vars[layer_name], target,
                    layer_name, self._pupil_dims['output_layers'][layer_idx])
                layer_name = 'output_layer_%s' % layer_idx
                outs[layer_name]['o_c'] = o
                outs[layer_name]['sigma_c'] = sigma
            return outs

    def _optimizer_core(self, optimizer_ins, state, gpu_idx, permute=True):
        if permute:
            optimizer_ins = self._extend_with_permutations(optimizer_ins, gpu_idx)
        if permute:
            optimizer_ins = self._forward_permute(optimizer_ins, ['o'], ['sigma'])
        optimizer_ins, num_concatenated = self._concat_opt_ins(optimizer_ins, ['o', 'sigma'])
        for idx, vars in enumerate(self._opt_trainable):
            optimizer_ins = self._apply_layer(
                optimizer_ins, vars, 'layer_%s' % idx)

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
            num_layers=4,
            num_gpus=1,
            regularization_rate=6e-6,
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
            no_end=False,
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
        self._num_layers = num_layers
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

        self._additional_metrics = additional_metrics

        self._flags = flags
        self._get_theta = get_theta
        self._get_omega_and_beta = get_omega_and_beta
        self._matrix_mod = matrix_mod
        self._no_end = no_end

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

            self._hooks['reset_permutation_matrices'] = tf.group(
                *self._reset_all_permutation_matrices())
        elif self._regime == 'inference':
            self._inference_graph()
            self._hooks['reset_optimizer_inference_state'] = self._empty_op

    def get_default_hooks(self):
        return construct_dict_without_none_entries(self._hooks)
