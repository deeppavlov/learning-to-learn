import numpy as np
import tensorflow as tf

from learning_to_learn.tensors import compute_metrics
from learning_to_learn.useful_functions import InvalidArgumentError, custom_matmul, custom_add, cumulative_mul, \
    construct_dict_without_none_entries
from learning_to_learn.pupils.pupil import Pupil


class ConvForMeta(Pupil):
    _name = 'mlp'

    @classmethod
    def get_name(cls):
        return cls._name

    @classmethod
    def check_kwargs(cls,
                     **kwargs):
        pass

    @staticmethod
    def create_saver(var_dict):
        # print("(Lstm.create_saver)var_dict:", var_dict)
        with tf.device('/cpu:0'):
            saved_vars = dict()
            for layer_idx, matrix in enumerate(var_dict['matrices']):
                saved_vars['matrix_%s' % layer_idx] = matrix
                saved_vars['bias_%s' % layer_idx] = var_dict['biases'][layer_idx]
            saver = tf.train.Saver(saved_vars, max_to_keep=None)
        return saver

    def _l2_loss(self, matrices):
        with tf.name_scope('l2_loss'):
            regularizer = tf.contrib.layers.l2_regularizer(self._regularization_rate)
            loss = 0
            for matr in matrices:
                loss += regularizer(matr)
            return loss

    def apply_mods(self, mods):
        assign_ops = list()
        for layer_idx, (matr, bias) in enumerate(
                zip(self._trainable['matrices'],
                    self._trainable['biases'])):
            assign_ops.append(
                tf.assign(
                    matr,
                    mods['layer_%s' % layer_idx]['matrix']
                )
            )
            mod = mods['layer_%s' % layer_idx]['bias']
            if len(mod.get_shape().as_list()) > 1:
                mod = tf.reshape(mod, [-1], name='bias_mod_first_dims_collapsed')
            assign_ops.append(
                tf.assign(
                    bias,
                    mod
                )
            )
        return assign_ops

    def _extract_trainable_from_opt_ins(self, opt_ins):
        trainable = dict(
            matrices=[opt_ins['layer_%s' % layer_idx]['matrix'] for layer_idx in range(self._num_layers)],
            biases=[opt_ins['layer_%s' % layer_idx]['bias'] for layer_idx in range(self._num_layers)],
        )
        return trainable

    def _acomplish_optimizer_ins(self, optimizer_ins, trainable_variables):
        for layer_idx in range(self._num_layers):
            optimizer_ins['layer_%s' % layer_idx]['matrix'] = trainable_variables['matrices'][layer_idx]
            optimizer_ins['layer_%s' % layer_idx]['bias'] = trainable_variables['biases'][layer_idx]
        return optimizer_ins

    def _mlp(self, inputs, matrices, biases):
        opt_ins = dict()
        inp_shape = inputs.get_shape().as_list()
        inp_shape = [-1 if a is None else a for a in inp_shape]
        ndim = len(inp_shape)
        inputs = tf.reshape(inputs, inp_shape[:ndim - self._input_ndim] + [self._input_size], name='inp_reshaped')
        hs = inputs
        with tf.name_scope('mlp'):
            for idx, (m, b) in enumerate(zip(matrices, biases)):
                layer_name = 'layer_%s' % idx
                opt_ins[layer_name] = dict(
                    o=hs
                )
                with tf.name_scope(layer_name):
                    preactivate = custom_add(custom_matmul(hs, m), b)
                    if idx < len(matrices) - 1:
                        hs = tf.nn.relu(preactivate)
                    else:
                        hs = preactivate
                opt_ins[layer_name]['s'] = preactivate
        return hs, opt_ins

    def loss_and_opt_ins(
            self, inputs, labels, opt_ins=None, trainable_variables=None, name_scope='pupil_loss'
    ):
        """Args:
            either trainable_variables or opt_ins have to be provided
        optimizer_ins is a dictionary which keys are layer names and values are dictionaries with their parameters
         ('matrix' and optionally 'bias') (meaning tf.Variable instances holding their saved values) and 'o' ansd 's'
        vectors. 'o' - vector is an output of previous layer (input for linear projection) and 's' is the result of
        linear projection performed using layer weights. 'o', 's', 'matrix', 'bias' - can be stacked if several
        exercises are processed"""
        with tf.name_scope(name_scope):
            if opt_ins is not None:
                trainable = self._extract_trainable_from_opt_ins(opt_ins)
            elif trainable_variables is not None:
                trainable = trainable_variables
            else:
                raise InvalidArgumentError(
                    'At least one of arguments opt_ins or trainable_variables have to be provided',
                    (None, None),
                    ('opt_ins', 'trainable_variables'),
                    'opt_ins in optimizer_ins format and trainable_variables structure is explained in loss_and_opt_ins'
                    'implementation'
                )

            matrices = trainable['matrices']
            biases = trainable['biases']

            logits, opt_ins = self._mlp(inputs, matrices, biases)
            self._logits = logits
            # with tf.device('/cpu:0'):
            #     logits = tf.Print(
            #         logits,
            #         [logits],
            #         message="\n(MlpForMeta.loss_and_opt_ins)logits:",
            #         summarize=320
            #     )
            #     logits = tf.Print(
            #         logits,
            #         [inputs],
            #         message="\n(MlpForMeta.loss_and_opt_ins)inputs:",
            #         summarize=320
            #     )
            predictions = tf.nn.softmax(logits)

            # with tf.device('/cpu:0'):
            #     predictions = tf.Print(
            #         predictions,
            #         [predictions],
            #         message="\n(MlpForMeta.loss_and_opt_ins)predictions:",
            #         summarize=320
            #     )
            # print('(LstmForMeta.loss_and_opt_ins)labels_shape:', labels.get_shape().as_list())
            # print('(LstmForMeta.loss_and_opt_ins)logits_shape:', logits.get_shape().as_list())
            labels = tf.one_hot(labels, self._num_classes)
            labels_shape = [-1 if a is None else a for a in labels.get_shape().as_list()]
            sce = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits
            )
            sce = tf.reshape(sce, labels_shape[:-1])
            loss = tf.reduce_mean(
                sce,
                axis=[-1]
            )
            optimizer_ins = self._acomplish_optimizer_ins(opt_ins, trainable)
            return loss, optimizer_ins, [], predictions, labels

    def loss_and_opt_ins_for_inference(self):
        loss, optimizer_ins, stor_save_ops, predictions, labels = self.loss_and_opt_ins(
            self._inputs_and_labels_placeholders['inputs'],
            self._inputs_and_labels_placeholders['labels'],
            trainable_variables=self._trainable
        )
        return loss, optimizer_ins, stor_save_ops, predictions, labels

    def _add_metrics_and_hooks(
            self,
            loss,
            predictions,
            labels
    ):
        self._hooks['loss'] = loss
        self._hooks['validation_loss'] = loss
        metrics = compute_metrics(
            self._additional_metrics,
            predictions=predictions,
            labels=labels,
            loss=loss,
            keep_first_dim=False
        )
        self._hooks['predictions'] = predictions
        self._hooks['validation_predictions'] = predictions
        for k, v in metrics.items():
            self._hooks[k] = v
            self._hooks['validation_' + k] = v

    def _add_train_op(
            self,
            loss
    ):
        # with tf.device('/cpu:0'):
        #     self._autonomous_train_specific_placeholders['learning_rate'] = tf.Print(
        #         self._autonomous_train_specific_placeholders['learning_rate'],
        #         [self._autonomous_train_specific_placeholders['learning_rate']],
        #         message="\n(MlpForMeta._add_train_op)self._autonomous_train_specific_placeholders['learning_rate']:",
        #         summarize=320
        #     )
        #     self._autonomous_train_specific_placeholders['learning_rate'] = tf.Print(
        #         self._autonomous_train_specific_placeholders['learning_rate'],
        #         [self._trainable['matrices'][0]],
        #         message="\n(MlpForMeta._add_train_op)self._trainable['matrices'][0]:",
        #         summarize=320
        #     )
        #     self._autonomous_train_specific_placeholders['learning_rate'] = tf.Print(
        #         self._autonomous_train_specific_placeholders['learning_rate'],
        #         [tf.gradients(loss, self._logits)],
        #         message="\n(MlpForMeta._add_train_op)sigma:",
        #         summarize=320
        #     )
        #     # loss = tf.Print(
        #     #     loss,
        #     #     [self._trainable['matrices'][1]],
        #     #     message="\n(MlpForMeta._add_train_op)self._trainable['matrices'][1]:",
        #     #     summarize=320
        #     # )
        if self._optimizer == 'adam':
            opt = tf.train.AdamOptimizer(
                learning_rate=self._autonomous_train_specific_placeholders['learning_rate'])
        elif self._optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(
                learning_rate=self._autonomous_train_specific_placeholders['learning_rate'])
        elif self._optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(
                learning_rate=self._autonomous_train_specific_placeholders['learning_rate'])
        elif self._optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(
                learning_rate=self._autonomous_train_specific_placeholders['learning_rate'])
        elif self._optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(
                learning_rate=self._autonomous_train_specific_placeholders['learning_rate'],
                momentum=self._autonomous_train_specific_placeholders['momentum']
            )
        elif self._optimizer == 'nesterov':
            opt = tf.train.MomentumOptimizer(
                learning_rate=self._autonomous_train_specific_placeholders['learning_rate'],
                momentum=self._autonomous_train_specific_placeholders['momentum'],
                use_nesterov=True
            )
        else:
            print('using sgd optimizer')
            opt = tf.train.GradientDescentOptimizer(
                self._autonomous_train_specific_placeholders['learning_rate'])
        l2_loss = self._l2_loss(self._trainable['matrices'])
        gv = opt.compute_gradients(loss + l2_loss)
        g, v = zip(*gv)
        g = list(g)
        # with tf.device('/cpu:0'):
        #     g[0] = tf.Print(
        #         g[0],
        #         g,
        #         message="(MlpForMeta._add_train_op)gradients:",
        #         summarize=320
        #     )
        gv = list(zip(g, v))
        self._hooks['train_op'] = opt.apply_gradients(gv)
        # self._hooks['train_op'] = opt.minimize(loss + l2_loss)

    def _compute_matrix_parameters(self, idx):
        if idx == 0:
            # print(self._num_hidden_nodes)
            input_dim = sum(self._input_shape)
        else:
            input_dim = self._num_hidden_nodes[idx - 1]
        if idx < self._num_layers - 1:
            output_dim = self._num_hidden_nodes[idx]
        else:
            output_dim = self._num_classes
        stddev = self._init_parameter * np.sqrt(1. / (input_dim + output_dim))
        return input_dim, output_dim, stddev

    def _create_trainable_variables_dictionary(self, device, name_scope):
        variables_dictionary = dict()
        with tf.device(device):
            with tf.name_scope(name_scope):
                matrices = list()
                biases = list()
                for layer_idx in range(self._num_layers):
                    input_dim, output_dim, stddev = self._compute_matrix_parameters(layer_idx)
                    matrices.append(
                        tf.Variable(tf.truncated_normal([input_dim,
                                                         output_dim],
                                                        stddev=stddev),
                                    name='matrix_%s' % layer_idx))
                    biases.append(tf.Variable(tf.zeros([output_dim]), name='bias_%s' % layer_idx))
                variables_dictionary['matrices'] = matrices
                variables_dictionary['biases'] = biases
        return variables_dictionary

    def _add_trainable_variables(self):
        trainable = self._trainable
        var_dict = self._create_trainable_variables_dictionary('/gpu:0', 'applicable_trainable')
        for k, v in var_dict.items():
            trainable[k] = v
        self._hooks['saver'] = self.create_saver(trainable)

    def _pack_trainable_to_optimizer_format(self, trainable):
        # print("(Lstm._pack_trainable_to_optimizer_format)trainable:", trainable)
        opt_ins = dict()
        for i in range(self._num_layers):
            opt_ins['layer_%s' % i] = dict(
                matrix=trainable['matrices'][i],
                bias=trainable['biases'][i]
            )
        return opt_ins

    def create_trainable_variables_dictionary_for_optimizer(self, device, name_scope):
        """Returns variable dictionary in 2 formats: for optimizer and for pupil"""
        variables_dictionary = self._create_trainable_variables_dictionary(device, name_scope)
        return self._pack_trainable_to_optimizer_format(variables_dictionary), variables_dictionary

    def make_inputs_and_labels_placeholders(self, device, name_scope):
        placeholders = dict()
        with tf.device(device):
            # shape_batch_dim = [self._batch_size]
            shape_batch_dim = [None]
            if name_scope is not None:
                with tf.name_scope(name_scope):
                    placeholders['inputs'] = tf.placeholder(
                        tf.float32, shape=shape_batch_dim + self._input_shape, name='inputs')
                    placeholders['labels'] = tf.placeholder(
                        tf.int32, shape=shape_batch_dim, name='labels')
            else:
                placeholders['inputs'] = tf.placeholder(
                    tf.float32, shape=shape_batch_dim + self._input_shape, name='inputs')
                placeholders['labels'] = tf.placeholder(
                    tf.int32, shape=shape_batch_dim, name='labels')
        return placeholders

    def _add_inputs_and_labels(self):
        placeholders = self.make_inputs_and_labels_placeholders('/gpu:0', 'applicable_placeholders')
        for k, v in placeholders.items():
            self._inputs_and_labels_placeholders[k] = v
        self._hooks['inputs'] = placeholders['inputs']
        self._hooks['labels'] = placeholders['labels']
        self._hooks['validation_inputs'] = placeholders['inputs']
        self._hooks['validation_labels'] = placeholders['labels']

    def _add_autonomous_training_specific_placeholders(self):
        with tf.device('/gpu:0'):
            self._autonomous_train_specific_placeholders['learning_rate'] = tf.placeholder(
                tf.float32, name='learning_rate')
            self._hooks['learning_rate'] = self._autonomous_train_specific_placeholders['learning_rate']
            if self._optimizer in ['momentum', 'nesterov']:
                self._autonomous_train_specific_placeholders['momentum'] = tf.placeholder(
                    tf.float32, name='momentum')
                self._hooks['momentum'] = self._autonomous_train_specific_placeholders['momentum']

    def __init__(
            self,
            batch_size=32,
            num_layers=2,
            num_channels=None,  # a list which length is by 1 less than num_layers
            init_parameter=3.,
            regularization_rate=.000006,
            additional_metrics=None,
            input_shape=None,
            num_classes=None,
            regime='autonomous_training',
            optimizer='adam'
    ):
        if num_hidden_nodes is None:
            num_hidden_nodes = [1000, 1000]

        self._batch_size = batch_size
        self._num_layers = num_layers
        self._num_hidden_nodes = num_hidden_nodes
        self._init_parameter = init_parameter
        self._regularization_rate = regularization_rate
        self._input_shape = input_shape
        self._input_size = cumulative_mul(input_shape, 1)
        self._input_ndim = len(self._input_shape)
        self._num_classes = num_classes
        self._additional_metrics = additional_metrics
        self._optimizer = optimizer

        self._hooks = dict(
            inputs=None,
            labels=None,
            labels_prepared=None,
            train_op=None,
            learning_rate=None,
            momentum=None,
            loss=None,
            predictions=None,
            validation_inputs=None,
            validation_labels=None,
            validation_predictions=None,
            dropout=None,
            saver=None
        )
        for add_metric in self._additional_metrics:
            self._hooks[add_metric] = None
            self._hooks['validation_' + add_metric] = None
        self._dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self._hooks['dropout'] = self._dropout_keep_prob

        self._trainable = dict()
        self._inputs_and_labels_placeholders = dict()
        self._autonomous_train_specific_placeholders = dict()

        self._add_trainable_variables()
        self._add_inputs_and_labels()
        loss, optimizer_ins, _, predictions, labels = self.loss_and_opt_ins_for_inference()
        self._add_metrics_and_hooks(loss, predictions, labels)
        if regime == 'autonomous_training':
            self._add_autonomous_training_specific_placeholders()
            self._add_train_op(loss)

        elif regime == 'inference':
            pass

        elif regime == 'training_with_meta_optimizer':
            pass

        elif regime == 'optimizer_training':
            pass

        else:
            raise InvalidArgumentError(
                'Not allowed regime',
                regime,
                'regime',
                ['autonomous_training', 'inference', 'training_with_meta_optimizer', 'optimizer_training']
            )

    def get_default_hooks(self):
        return construct_dict_without_none_entries(self._hooks)

    def get_building_parameters(self):
        pass

    def get_net_size(self):
        return dict(
            num_nodes=self._num_hidden_nodes,
            num_layers=self._num_layers,
            batch_size=self._batch_size,
            input_shape=self._input_shape,
            input_ndim=self._input_ndim,
            input_size=self._input_size
        )

    def get_layer_dims(self):
        dims = dict()
        dims['layers'] = []
        for layer_idx, _ in enumerate(self._num_hidden_nodes):
            in_dim, out_dim, _ = self._compute_matrix_parameters(layer_idx)
            dims['layers'].append((in_dim, out_dim))
        return dims
