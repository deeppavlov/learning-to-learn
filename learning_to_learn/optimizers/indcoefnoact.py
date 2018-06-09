from itertools import chain

import tensorflow as tf

from learning_to_learn.optimizers.meta import Meta
from learning_to_learn.useful_functions import flatten, construct_dict_without_none_entries, unite_nested_dicts


class IndCoefNoAct(Meta):

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
        return list()

    def _create_permutation_matrices(self, num_exercises, gpu_idx):
        pass

    @staticmethod
    def _reset_optimizer_states(var_scope, gpu_idx):
        return list()

    def _create_optimizer_trainable_vars(self):
        total_ndim = sum(flatten(self._pupil_dims)) * self._pupil_net_size['num_unrollings']
        with tf.variable_scope('optimizer_trainable_variables'):
            matrix = tf.Variable(tf.ones([total_ndim]), name='coefs')
            bias = tf.Variable(tf.zeros([total_ndim]), name='bias')
            vars = [matrix, bias]
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, matrix)
        return vars

    # def _optimizer_core(self, optimizer_ins, num_exercises, states, gpu_idx):
    #     # optimizer_ins = self._extend_with_permutations(optimizer_ins, num_exercises, gpu_idx)
    #     # optimizer_ins = self._forward_permute(optimizer_ins)
    #     return self._empty_core(optimizer_ins)

    def _apply_layer(self, inp, vars):
        with tf.name_scope('apply_coef'):
            return tf.multiply(inp, vars[0]) + vars[1]

    def _optimizer_core(self, optimizer_ins, state, gpu_idx, permute=True):
        vec, map_ = self._all_ins_2_1_vec(optimizer_ins, ['o', 'sigma'])
        output = self._apply_layer(vec, self._opt_trainable)
        outs = self._unpack_all_ins_from_1_vec(output, map_)
        outs = self._mv_tensors(outs, ['o', 'sigma'], ['o_pr', 'sigma_pr'])
        optimizer_outs = unite_nested_dicts([optimizer_ins, outs], 1)
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

        self._permute = False

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
