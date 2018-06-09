import random

import numpy as np
import tensorflow as tf
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary, char2vec, pred2vec, \
    pred2vec_fast, vec2char, vec2char_fast, char2id, id2char, get_available_gpus, device_name_scope, \
    average_gradients, get_num_gpus_and_bs_on_gpus, custom_matmul, custom_add, InvalidArgumentError, \
    compose_save_list, compose_reset_list, compose_randomize_list, construct_dict_without_none_entries, \
    append_to_nested, get_average_with_weights_func, func_on_list_in_nested

from learning_to_learn.tensors import compute_metrics


class LstmBatchGenerator(object):
    @staticmethod
    def create_vocabulary(texts):
        text = ''
        for t in texts:
            text += t
        return create_vocabulary(text)

    @staticmethod
    def char2vec(char, character_positions_in_vocabulary, speaker_idx, speaker_flag_size):
        return np.reshape(char2vec(char, character_positions_in_vocabulary), (1, 1, -1))

    @staticmethod
    def pred2vec(pred, speaker_idx, speaker_flag_size, batch_gen_args):
        return np.reshape(pred2vec(pred), (1, 1, -1))

    @staticmethod
    def vec2char(vec, vocabulary):
        return vec2char(vec, vocabulary)

    @staticmethod
    def vec2char_fast(vec, vocabulary):
        return vec2char(vec, vocabulary)

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None, random_batch_initiation=False):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabulary = vocabulary
        self._vocabulary_size = len(self.vocabulary)
        self.character_positions_in_vocabulary = get_positions_in_vocabulary(self.vocabulary)
        self._num_unrollings = num_unrollings
        if random_batch_initiation:
            self._cursor = random.sample(range(self._text_size), batch_size)
        else:
            segment = self._text_size // batch_size
            self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()

    def get_num_batches(self):
        return len(self._text)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    def _start_batch(self):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, 0] = 1.0
        return batch

    def _zero_batch(self):
        return np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]], self.character_positions_in_vocabulary)] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def char2batch(self, char):
        return np.stack(char2vec(char, self.character_positions_in_vocabulary)), np.stack(self._zero_batch())

    def pred2batch(self, pred):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        char_id = np.argmax(pred, 1)[-1]
        batch[0, char_id] = 1.0
        return np.stack([batch]), np.stack([self._zero_batch()])

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0)


class LstmFastBatchGenerator(object):
    @staticmethod
    def create_vocabulary(texts):
        text = ''
        for t in texts:
            text += t
        return create_vocabulary(text)

    @staticmethod
    def char2vec(char, character_positions_in_vocabulary, speaker_idx, speaker_flag_size):
        return np.array([char2id(char, character_positions_in_vocabulary)])

    @staticmethod
    def pred2vec(pred, speaker_idx, speaker_flag_size, batch_gen_args):
        return np.reshape(pred2vec_fast(pred), (1, -1, 1))

    @staticmethod
    def vec2char(vec, vocabulary):
        return vec2char(vec, vocabulary)

    @staticmethod
    def vec2char_fast(vec, vocabulary):
        return vec2char_fast(vec, vocabulary)

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None, random_batch_initiation=False):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabulary = vocabulary
        self._vocabulary_size = len(self.vocabulary)
        self.character_positions_in_vocabulary = get_positions_in_vocabulary(self.vocabulary)
        self._num_unrollings = num_unrollings
        if random_batch_initiation:
            self._cursor = random.sample(range(self._text_size), batch_size)
        else:
            segment = self._text_size // batch_size
            self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()

    def get_num_batches(self):
        return len(self._text)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    def _start_batch(self):
        return np.array([[0] for _ in range(self._batch_size)])

    def _zero_batch(self):
        return -np.ones(shape=(self._batch_size), dtype=np.float)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        ret = np.array([[char2id(self._text[self._cursor[b]], self.character_positions_in_vocabulary)]
                        for b in range(self._batch_size)])
        for b in range(self._batch_size):
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return ret

    def char2batch(self, char):
        return np.stack(char2vec(char, self.character_positions_in_vocabulary)), np.stack(self._zero_batch())

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        # print('(LstmFastBatchGenerator.next)batches[:-1]:', batches[:-1])
        # print('(LstmFastBatchGenerator.next)batches[:-1].shape:', [b.shape for b in batches[:-1]])
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0)


def characters(probabilities, vocabulary):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c, vocabulary) for c in np.argmax(probabilities, 1)]


def batches2string(batches, vocabulary):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [u""] * batches[0].shape[0]
    for b in batches:
        s = [u"".join(x) for x in zip(s, characters(b, vocabulary))]
    return s


class Model(object):
    @classmethod
    def get_name(cls):
        return cls.name


class Lstm(Model):
    _name = 'lstm'

    @classmethod
    def check_kwargs(cls,
                     **kwargs):
        pass

    @classmethod
    def get_name(cls):
        return cls._name

    def get_special_args(self):
        return dict()

    @staticmethod
    def form_kwargs(kwargs_for_building, insertions):
        for insertion in insertions:
            if insertion['list_index'] is None:
                kwargs_for_building[insertion['hp_name']] = insertion['paste']
            else:
                kwargs_for_building[insertion['hp_name']][insertion['list_index']] = insertion['paste']
        return kwargs_for_building

    def _lstm_layer(self, inp, state, layer_idx, matr, bias):
        with tf.name_scope('lstm_layer_%s' % layer_idx):
            nn = self._num_nodes[layer_idx]
            x = tf.concat(
                [tf.nn.dropout(
                    inp,
                    self._dropout_keep_prob),
                 state[0]],
                -1,
                name='X')
            s = custom_matmul(x, matr)
            linear_res = custom_add(
                s, bias, name='linear_res')
            [sigm_arg, tanh_arg] = tf.split(linear_res, [3 * nn, nn], axis=-1, name='split_to_act_func_args')
            sigm_res = tf.sigmoid(sigm_arg, name='sigm_res')
            transform_vec = tf.tanh(tanh_arg, name='transformation_vector')
            [forget_gate, input_gate, output_gate] = tf.split(sigm_res, 3, axis=-1, name='gates')
            new_cell_state = tf.add(forget_gate * state[1], input_gate * transform_vec, name='new_cell_state')
            new_hidden_state = tf.multiply(output_gate, tf.tanh(new_cell_state), name='new_hidden_state')
        optimizer_ins = {'lstm_layer_%s' % layer_idx: {'o': x, 's': s}}
        return new_hidden_state, [new_hidden_state, new_cell_state], optimizer_ins

    def _rnn_iter(self, embedding, all_states, lstm_matrices, lstm_biases):
        optimizer_ins = dict()
        with tf.name_scope('rnn_iter'):
            new_all_states = list()
            output = embedding
            for layer_idx, state in enumerate(all_states):
                output, state, opt_ins = self._lstm_layer(
                    output, state, layer_idx, lstm_matrices[layer_idx], lstm_biases[layer_idx])
                new_all_states.append(state)
                optimizer_ins.update(opt_ins)
            return output, new_all_states, optimizer_ins

    def _rnn_module(self, embeddings, all_states, lstm_matrices, lstm_biases):
        rnn_outputs = list()
        optimizer_ins = dict()
        for layer_idx in range(self._num_layers):
            optimizer_ins['lstm_layer_%s' % layer_idx] = {'o': list(), 's': list()}
        with tf.name_scope('rnn_module'):
            for emb in embeddings:
                rnn_output, all_states, opt_ins = self._rnn_iter(emb, all_states, lstm_matrices, lstm_biases)
                # print('rnn_output.shape:', rnn_output.get_shape().as_list())
                for layer_idx in range(self._num_layers):
                    optimizer_ins['lstm_layer_%s' % layer_idx]['o'].append(opt_ins['lstm_layer_%s' % layer_idx]['o'])
                    optimizer_ins['lstm_layer_%s' % layer_idx]['s'].append(opt_ins['lstm_layer_%s' % layer_idx]['s'])
                rnn_outputs.append(rnn_output)
        return rnn_outputs, all_states, optimizer_ins

    @staticmethod
    def _embed(inputs, matrix):
        with tf.name_scope('embeddings'):
            embeddings = custom_matmul(inputs, matrix, base_ndims=[3, 2])
            embeddings_ndims = len(embeddings.get_shape().as_list())
            if embeddings_ndims == 4:
                unstack_dim = 1
            else:
                unstack_dim = 0
            unstacked_embeddings = tf.unstack(embeddings, axis=unstack_dim, name='embeddings')
            optimizer_ins = {'embedding_layer': {'o': tf.unstack(inputs, axis=unstack_dim, name='embeddings'),
                                                 's': unstacked_embeddings}}
            return unstacked_embeddings, optimizer_ins

    def _output_module(self, rnn_outputs, output_matrices, output_biases):
        optimizer_ins = dict()
        with tf.name_scope('output_module'):
            # print('rnn_outputs:', rnn_outputs)
            rnn_output_ndim = len(rnn_outputs[0].get_shape().as_list())
            if rnn_output_ndim == 3:
                concat_dim = 1
            else:
                concat_dim = 0
            num_split = len(rnn_outputs)
            # o = rnn_outputs
            # rnn_outputs = tf.concat(rnn_outputs, concat_dim, name='concatenated_rnn_outputs')
            o = rnn_outputs
            for layer_idx, (matr, bias) in enumerate(zip(output_matrices, output_biases)):
                with tf.name_scope('layer_%s' % layer_idx):
                    # print('hs.shape:', hs.get_shape().as_list())
                    if layer_idx > 0:
                        o = tf.split(hs, num_split, axis=concat_dim, name='split_o_for_optimizer')
                    hs = tf.concat(o, concat_dim, name='united_o')
                    s = custom_matmul(
                            hs, matr, name_scope='first_s')
                    s = tf.split(s, num_split, axis=concat_dim, name='split_s_for_optimizer')
                    optimizer_ins['output_layer_%s' % layer_idx] = dict(
                        o=o,
                        s=s
                    )
                    s = tf.concat(s, concat_dim, name='united_s')
                    hs = custom_add(
                        s,
                        bias,
                        name='res_of_%s_output_layer' % layer_idx)
                    if layer_idx < self._num_output_layers - 1:
                        hs = tf.nn.relu(hs)
        return hs, optimizer_ins

    def _compute_lstm_matrix_parameters(self, idx):
        if idx == 0:
            # print(self._num_nodes)
            input_dim = self._num_nodes[0] + self._embedding_size
        else:
            input_dim = self._num_nodes[idx - 1] + self._num_nodes[idx]
        output_dim = 4 * self._num_nodes[idx]
        stddev = self._init_parameter * np.sqrt(1. / input_dim)
        return input_dim, output_dim, stddev

    def _compute_output_matrix_parameters(self, idx):
        if idx == 0:
            # print('self._num_nodes:', self._num_nodes)
            input_dim = self._num_nodes[-1]
        else:
            input_dim = self._num_output_nodes[idx - 1]
        if idx == self._num_output_layers - 1:
            output_dim = self._vec_dim
        else:
            output_dim = self._num_output_nodes[idx]
        stddev = self._init_parameter * np.sqrt(1. / input_dim)
        return input_dim, output_dim, stddev

    def _l2_loss(self, matrices):
        with tf.name_scope('l2_loss'):
            regularizer = tf.contrib.layers.l2_regularizer(self._regularization_rate)
            loss = 0
            for matr in matrices:
                loss += regularizer(matr)
            return loss

    def _acomplish_optimizer_ins(self, optimizer_ins, trainable_variables):
        optimizer_ins['embedding_layer']['matrix'] = trainable_variables['embedding_matrix']
        for layer_idx in range(self._num_layers):
            optimizer_ins['lstm_layer_%s' % layer_idx]['matrix'] = trainable_variables['lstm_matrices'][layer_idx]
            optimizer_ins['lstm_layer_%s' % layer_idx]['bias'] = trainable_variables['lstm_biases'][layer_idx]
        for layer_idx in range(self._num_output_layers):
            optimizer_ins['output_layer_%s' % layer_idx]['matrix'] = \
                trainable_variables['output_matrices'][layer_idx]
            optimizer_ins['output_layer_%s' % layer_idx]['bias'] = trainable_variables['output_biases'][layer_idx]
        return optimizer_ins

    def _extract_trainable_from_opt_ins(self, opt_ins):
        trainable = dict(
            embedding_matrix=opt_ins['embedding_layer']['matrix'],
            lstm_matrices=[opt_ins['lstm_layer_%s' % layer_idx]['matrix'] for layer_idx in range(self._num_layers)],
            lstm_biases=[opt_ins['lstm_layer_%s' % layer_idx]['bias'] for layer_idx in range(self._num_layers)],
            output_matrices=[opt_ins['output_layer_%s' % layer_idx]['matrix']
                             for layer_idx in range(self._num_output_layers)],
            output_biases=[opt_ins['output_layer_%s' % layer_idx]['bias']
                             for layer_idx in range(self._num_output_layers)]
        )
        return trainable

    def loss_and_opt_ins(
            self, inputs, labels, storage, opt_ins=None, trainable_variables=None, name_scope='pupil_loss'):
        """Args:
            either trainable_variables or opt_ins have to be provided
        optimizer_ins is a dictionary which keys are layer names and values are dictionaries with their parameters
         ('matrix' and optionally 'bias') (meaning tf.Variable instances holding their saved values) and 'o' ansd 's'
        vectors. 'o' - vector is an output of previous layer (input for linear projection) and 's' is the result of
        linear projection performed using layer weights. 'o', 's', 'matrix', 'bias' - can be stacked if several
        exercises are processed"""
        optimizer_ins = dict()
        with tf.name_scope(name_scope):
            saved_states = storage['states']
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

            embedding_matrix = trainable['embedding_matrix']
            lstm_matrices = trainable['lstm_matrices']
            lstm_biases = trainable['lstm_biases']
            output_matrices = trainable['output_matrices']
            output_biases = trainable['output_biases']

            # with tf.device('/cpu:0'):
            #     embedding_matrix = tf.Print(
            #         embedding_matrix, [embedding_matrix], message="\nembedding_matrix: ", summarize=10)
            #     n_lstm_matrices = list()
            #     for lstm_matrix in lstm_matrices:
            #         n_lstm_matrices.append(tf.Print(
            #             lstm_matrix, [lstm_matrix], message="lstm_matrix: ", summarize=10))
            #     n_output_matrices = list()
            #     for output_matrix in output_matrices:
            #         n_output_matrices.append(tf.Print(
            #             output_matrix, [output_matrix], message="output_matrix: ", summarize=10))
            #     lstm_matrices = n_lstm_matrices
            #     output_matrices = n_output_matrices
            #     n_saved_states = list()
            #     for layer_saved_states in saved_states:
            #         n_saved_states.append([
            #             tf.Print(layer_saved_states[0], [layer_saved_states[0]], message="h: ", summarize=10),
            #             tf.Print(layer_saved_states[1], [layer_saved_states[1]], message="c: ", summarize=10)
            #         ])
            #     saved_states = n_saved_states

            inputs, labels = self._prepare_inputs_and_labels(inputs, labels)

            embeddings, opt_ins = self._embed(inputs, embedding_matrix)
            optimizer_ins.update(opt_ins)
            rnn_outputs, new_states, opt_ins = self._rnn_module(
                embeddings, saved_states, lstm_matrices, lstm_biases)
            optimizer_ins.update(opt_ins)
            logits, opt_ins = self._output_module(rnn_outputs, output_matrices, output_biases)
            optimizer_ins.update(opt_ins)

            new_storage = dict(
                states=new_states
            )
            labels_shape = labels.get_shape().as_list()
            logits = tf.reshape(logits, labels_shape),
            predictions = tf.nn.softmax(logits)
            # print('(LstmForMeta.loss_and_opt_ins)labels_shape:', labels.get_shape().as_list())
            # print('(LstmForMeta.loss_and_opt_ins)logits_shape:', logits.get_shape().as_list())
            sce = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.reshape(labels, [-1, labels_shape[-1]]),
                logits=tf.reshape(logits, [-1, labels_shape[-1]])
            )
            loss = tf.reduce_mean(
                tf.reshape(sce, labels_shape[:-1]),
                axis=[-1, -2]
            )
            optimizer_ins = self._acomplish_optimizer_ins(optimizer_ins, trainable)
            return loss, optimizer_ins, new_storage, predictions, labels

    def loss_and_opt_ins_for_inference(self):
        loss, optimizer_ins, new_storage, predictions, labels = self.loss_and_opt_ins(
            self._train_inputs_and_labels_placeholders['inputs'],
            self._train_inputs_and_labels_placeholders['labels'],
            self._train_storage,
            trainable_variables=self._applicable_trainable
        )
        # print('(Lstm.loss_and_opt_ins_for_inference)self._train_storage:', self._train_storage)
        # print('(Lstm.loss_and_opt_ins_for_inference)new_storage:', new_storage)
        storage_save_ops = compose_save_list(
            (self._train_storage, new_storage),
            name_scope='save_pupil_train_storage'
        )
        # print('(Lstm.loss_and_opt_ins_for_inference)storage_save_ops:', storage_save_ops)
        return loss, optimizer_ins, storage_save_ops, predictions, labels

    def apply_mods(self, mods):
        assign_ops = list()
        assign_ops.append(
            tf.assign(
                self._applicable_trainable['embedding_matrix'],
                mods['embedding_layer']['matrix']))
        for layer_idx, (matr, bias) in enumerate(
                zip(self._applicable_trainable['lstm_matrices'],
                    self._applicable_trainable['lstm_biases'])):
            assign_ops.append(
                tf.assign(
                    matr,
                    mods['lstm_layer_%s' % layer_idx]['matrix']
                )
            )
            mod = mods['lstm_layer_%s' % layer_idx]['bias']
            if len(mod.get_shape().as_list()) > 1:
                mod = tf.reshape(mod, [-1], name='bias_mod_first_dims_collapsed')
            assign_ops.append(
                tf.assign(
                    bias,
                    mod
                )
            )
        for layer_idx, (matr, bias) in enumerate(
                zip(self._applicable_trainable['output_matrices'],
                    self._applicable_trainable['output_biases'])):
            assign_ops.append(
                tf.assign(
                    matr,
                    mods['output_layer_%s' % layer_idx]['matrix']
                )
            )
            mod = mods['output_layer_%s' % layer_idx]['bias']
            if len(mod.get_shape().as_list()) > 1:
                mod = tf.reshape(mod, [-1], name='bias_mod_first_dims_collapsed')
            assign_ops.append(
                tf.assign(
                    bias,
                    mod
                )
            )
        return assign_ops

    def _train_graph(self):
        inputs, labels = self._prepare_inputs_and_labels(
            self._train_inputs_and_labels_placeholders['inputs'], self._train_inputs_and_labels_placeholders['labels'])
        inputs_by_device, labels_by_device = self._distribute_by_gpus(inputs, labels)
        trainable = self._applicable_trainable
        tower_grads = list()
        preds = list()
        losses = list()
        reset_state_ops = list()

        additional_metrics = dict()
        for add_metric in self._additional_metrics:
            additional_metrics[add_metric] = list()

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
        with tf.name_scope('train'):
            for gpu_batch_size, gpu_name, device_inputs, device_labels in zip(
                    self._batch_sizes_on_gpus, self._gpu_names, inputs_by_device, labels_by_device):
                with tf.device(gpu_name):
                    with tf.name_scope(device_name_scope(gpu_name)):
                        saved_states = list()
                        for layer_idx, layer_num_nodes in enumerate(self._num_nodes):
                            saved_states.append(
                                (tf.Variable(
                                    tf.zeros([gpu_batch_size, layer_num_nodes]),
                                    trainable=False,
                                    name='saved_state_%s_%s' % (layer_idx, 0)),
                                 tf.Variable(
                                     tf.zeros([gpu_batch_size, layer_num_nodes]),
                                     trainable=False,
                                     name='saved_state_%s_%s' % (layer_idx, 1)))
                            )
                        reset_state_ops.extend(compose_reset_list(saved_states))

                        all_states = saved_states
                        embeddings, _ = self._embed(device_inputs, trainable['embedding_matrix'])
                        rnn_outputs, all_states, _ = self._rnn_module(
                            embeddings, all_states, trainable['lstm_matrices'],
                            trainable['lstm_biases'])
                        logits, _ = self._output_module(
                            rnn_outputs, trainable['output_matrices'], trainable['output_biases'])

                        save_ops = compose_save_list((saved_states, all_states))

                        with tf.control_dependencies(save_ops):
                            all_matrices = [trainable['embedding_matrix']]
                            all_matrices.extend(trainable['lstm_matrices'])
                            all_matrices.extend(trainable['output_matrices'])
                            l2_loss = self._l2_loss(all_matrices)

                            loss = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits(labels=device_labels, logits=logits))
                            concat_pred = tf.nn.softmax(logits)

                            add_metrics = compute_metrics(
                                additional_metrics, predictions=concat_pred,
                                labels=device_labels, loss=loss, keep_first_dim=False
                            )

                            preds.append(tf.split(concat_pred, self._num_unrollings))

                            losses.append(loss)

                            grads_and_vars = opt.compute_gradients(loss + l2_loss)
                            tower_grads.append(grads_and_vars)
                            additional_metrics = append_to_nested(additional_metrics, add_metrics)

                            # splitting concatenated results for different characters
            with tf.device(self._base_device):
                with tf.name_scope(device_name_scope(self._base_device) + '_gradients'):
                    # print('(Lstm._train_graph)tower_grads:', tower_grads)
                    grads_and_vars = average_gradients(tower_grads)
                    grads, v = zip(*grads_and_vars)
                    # grads, _ = tf.clip_by_global_norm(grads, 1.)
                    self.train_op = opt.apply_gradients(zip(grads, v))
                    self._hooks['train_op'] = self.train_op
                    self._hooks['reset_pupil_train_state'] = tf.group(*reset_state_ops)
                    # composing predictions
                    preds_by_char = list()
                    # print('preds:', preds)
                    for one_char_preds in zip(*preds):
                        # print('one_char_preds:', one_char_preds)
                        preds_by_char.append(tf.concat(one_char_preds, 0))
                    # print('len(preds_by_char):', len(preds_by_char))
                    self.predictions = tf.concat(preds_by_char, 0)
                    self._hooks['predictions'] = self.predictions
                    # print('self.predictions.get_shape().as_list():', self.predictions.get_shape().as_list())
                    average_func = get_average_with_weights_func(self._batch_sizes_on_gpus)
                    self.loss = average_func(losses)
                    additional_metrics = func_on_list_in_nested(additional_metrics, average_func)
                    self._hooks['loss'] = self.loss
                    for k, v in additional_metrics.items():
                        self._hooks[k] = v

    def _validation_graph(self):
        trainable = self._applicable_trainable
        with tf.device(self._gpu_names[0]):
            with tf.name_scope('validation'):
                self.validation_labels = tf.placeholder(tf.int32, [1, 1])
                self.sample_input = tf.placeholder(tf.int32,
                                                   shape=[1, 1, 1],
                                                   name='sample_input')
                inputs = tf.reshape(self.sample_input, [1, -1])
                sample_input = tf.one_hot(inputs, self._vocabulary_size)
                labels = tf.reshape(self.validation_labels, [1])
                validation_labels_prepared = tf.one_hot(labels, self._vocabulary_size)

                self._hooks['validation_inputs'] = self.sample_input
                self._hooks['validation_labels'] = self.validation_labels
                self._hooks['validation_labels_prepared'] = validation_labels_prepared
                saved_sample_state = list()
                for layer_idx, layer_num_nodes in enumerate(self._num_nodes):
                    saved_sample_state.append(
                        (tf.Variable(
                            tf.zeros([1, layer_num_nodes]),
                            trainable=False,
                            name='saved_sample_state_%s_%s' % (layer_idx, 0)),
                         tf.Variable(
                             tf.zeros([1, layer_num_nodes]),
                             trainable=False,
                             name='saved_sample_state_%s_%s' % (layer_idx, 1)))
                    )

                reset_list = compose_reset_list(saved_sample_state)

                self.reset_sample_state = tf.group(*reset_list)
                self._hooks['reset_validation_state'] = self.reset_sample_state

                randomize_list = compose_randomize_list(saved_sample_state)

                self.randomize = tf.group(*randomize_list)
                self._hooks['randomize_sample_state'] = self.randomize

                embeddings, _ = self._embed(sample_input, trainable['embedding_matrix'])
                # print('embeddings:', embeddings)
                rnn_output, sample_state, _ = self._rnn_module(
                    embeddings, saved_sample_state, trainable['lstm_matrices'],
                    trainable['lstm_biases'])
                sample_logit, _ = self._output_module(
                    rnn_output, trainable['output_matrices'], trainable['output_biases'])

                sample_save_ops = compose_save_list((saved_sample_state, sample_state))

                with tf.control_dependencies(sample_save_ops):
                    self.sample_prediction = tf.nn.softmax(sample_logit)
                    metrics = compute_metrics(
                        self._additional_metrics + ['loss'],
                        predictions=self.sample_prediction,
                        labels=validation_labels_prepared,
                        keep_first_dim=False
                    )

                    self._hooks['validation_predictions'] = self.sample_prediction
                    for k, v in metrics.items():
                        self._hooks['validation_' + k] = v

    def _pack_trainable_to_optimizer_format(self, trainable):
        # print("(Lstm._pack_trainable_to_optimizer_format)trainable:", trainable)
        opt_ins = dict()
        opt_ins['embedding_layer'] = dict(
            matrix=trainable['embedding_matrix']
        )
        for i in range(self._num_layers):
            opt_ins['lstm_layer_%s' % i] = dict(
                matrix=trainable['lstm_matrices'][i],
                bias=trainable['lstm_biases'][i]
            )
        for i in range(self._num_output_layers):
            opt_ins['output_layer_%s' % i] = dict(
                matrix=trainable['output_matrices'][i],
                bias=trainable['output_biases'][i]
            )
        return opt_ins

    def _create_trainable_variables_dictionary(self, device, name_scope):
        variables_dictionary = dict()
        with tf.device(device):
            with tf.name_scope(name_scope):
                embedding_matrix = tf.Variable(
                    tf.truncated_normal([self._vec_dim, self._embedding_size],
                                        stddev=self._init_parameter * np.sqrt(1. / self._vec_dim)),
                    name='embedding_matrix')
                lstm_matrices = list()
                lstm_biases = list()
                for layer_idx in range(self._num_layers):
                    input_dim, output_dim, stddev = self._compute_lstm_matrix_parameters(layer_idx)
                    lstm_matrices.append(
                        tf.Variable(tf.truncated_normal([input_dim,
                                                         output_dim],
                                                        stddev=stddev),
                                    name='lstm_matrix_%s' % layer_idx))
                    lstm_biases.append(tf.Variable(tf.zeros([output_dim]), name='lstm_bias_%s' % layer_idx))
                output_matrices = list()
                output_biases = list()
                for layer_idx in range(self._num_output_layers):
                    input_dim, output_dim, stddev = self._compute_output_matrix_parameters(layer_idx)
                    # print('input_dim:', input_dim)
                    # print('output_dim:', output_dim)
                    output_matrices.append(
                        tf.Variable(tf.truncated_normal([input_dim, output_dim],
                                                        stddev=stddev),
                                    name='output_matrix_%s' % layer_idx))
                    output_biases.append(tf.Variable(
                        tf.zeros([output_dim]),
                        name='output_bias_%s' % layer_idx))
                variables_dictionary['embedding_matrix'] = embedding_matrix
                variables_dictionary['lstm_matrices'] = lstm_matrices
                variables_dictionary['lstm_biases'] = lstm_biases
                variables_dictionary['output_matrices'] = output_matrices
                variables_dictionary['output_biases'] = output_biases
        return variables_dictionary

    def create_trainable_variables_dictionary_for_optimizer(self, device, name_scope):
        """Returns variable dictionary in 2 formats: for optimizer and for pupil"""
        variables_dictionary = self._create_trainable_variables_dictionary(device, name_scope)
        return self._pack_trainable_to_optimizer_format(variables_dictionary), variables_dictionary

    @staticmethod
    def create_saver(var_dict):
        # print("(Lstm.create_saver)var_dict:", var_dict)
        with tf.device('/cpu:0'):
            saved_vars = dict()
            saved_vars['embedding_matrix'] = var_dict['embedding_matrix']
            for layer_idx, lstm_matrix in enumerate(var_dict['lstm_matrices']):
                saved_vars['lstm_matrix_%s' % layer_idx] = lstm_matrix
                saved_vars['lstm_bias_%s' % layer_idx] = var_dict['lstm_biases'][layer_idx]
            for layer_idx, (output_matrix, output_bias) in \
                    enumerate(zip(var_dict['output_matrices'], var_dict['output_biases'])):
                saved_vars['output_matrix_%s' % layer_idx] = output_matrix
                saved_vars['output_bias_%s' % layer_idx] = output_bias
            saver = tf.train.Saver(saved_vars, max_to_keep=None)
        return saver

    def _add_trainable_variables(self):
        trainable = self._applicable_trainable
        var_dict = self._create_trainable_variables_dictionary(self._base_device, 'applicable_trainable')
        for k, v in var_dict.items():
            trainable[k] = v
        self._hooks['saver'] = self.create_saver(trainable)

    def create_storage(self, device, name_scope):
        storage_dictionary = dict()
        with tf.device(device):
            with tf.name_scope(name_scope):
                with tf.name_scope('states'):
                    storage_dictionary['states'] = list()
                    states = storage_dictionary['states']
                    for layer_idx, layer_num_nodes in enumerate(self._num_nodes):
                        states.append(
                            [tf.Variable(
                                 tf.zeros([self._batch_size, layer_num_nodes]),
                                 trainable=False,
                                 name='saved_state_%s_%s' % (layer_idx, 0)),
                             tf.Variable(
                                 tf.zeros([self._batch_size, layer_num_nodes]),
                                 trainable=False,
                                 name='saved_state_%s_%s' % (layer_idx, 1))]
                        )
        return storage_dictionary

    @staticmethod
    def reset_storage(storage):
        return compose_reset_list(storage['states'])

    def reset_self_train_storage(self):
        return self.reset_storage(self._train_storage)

    def _add_train_storage(self):
        storage = self.create_storage(self._base_device, 'train_storage')
        for k, v in storage.items():
            self._train_storage[k] = v

    def make_inputs_and_labels_placeholders(self, device, name_scope):
        placeholders = dict()
        with tf.device(device):
            if name_scope is not None:
                with tf.name_scope(name_scope):
                    placeholders['inputs'] = tf.placeholder(
                        tf.int32, shape=[self._num_unrollings, self._batch_size, 1], name='inputs')
                    placeholders['labels'] = tf.placeholder(
                        tf.int32, shape=[self._num_unrollings * self._batch_size, 1], name='labels')
            else:
                placeholders['inputs'] = tf.placeholder(
                    tf.int32, shape=[self._num_unrollings, self._batch_size, 1], name='inputs')
                placeholders['labels'] = tf.placeholder(
                    tf.int32, shape=[self._num_unrollings * self._batch_size, 1], name='labels')
        return placeholders

    def _add_train_inputs_and_labels_placeholders(self):
        placeholders = self.make_inputs_and_labels_placeholders(self._base_device, 'applicable_placeholders')
        for k, v in placeholders.items():
            self._train_inputs_and_labels_placeholders[k] = v
        self._hooks['inputs'] = placeholders['inputs']
        self._hooks['labels'] = placeholders['labels']

    def _add_autonomous_training_specific_placeholders(self):
        with tf.device(self._base_device):
            self._autonomous_train_specific_placeholders['learning_rate'] = tf.placeholder(
                tf.float32, name='learning_rate')
            self._hooks['learning_rate'] = self._autonomous_train_specific_placeholders['learning_rate']
            if self._optimizer in ['momentum', 'nesterov']:
                self._autonomous_train_specific_placeholders['momentum'] = tf.placeholder(
                    tf.float32, name='momentum')
                self._hooks['momentum'] = self._autonomous_train_specific_placeholders['momentum']

    def _prepare_inputs_and_labels(self, inputs, labels):
        with tf.device(self._base_device):
            """last dimension is excess and has size one"""
            inputs_shape = inputs.get_shape().as_list()
            batch_size = inputs_shape[-2]
            num_unrollings = inputs_shape[-3]
            ndims = len(inputs_shape)
            # print('(LstmForMeta._prepare_inputs_and_labels)inputs_shape:', inputs_shape)
            if ndims == 4:
                num_exercises = inputs_shape[0]
            else:
                num_exercises = None
            if ndims == 4:
                new_inp_shape = [num_exercises, num_unrollings, batch_size]
                prepared_lbls_shape = [num_exercises, num_unrollings*batch_size]
                lbls_for_loss_shape = [num_exercises, num_unrollings, batch_size, self._vec_dim]
            elif ndims == 3:
                prepared_lbls_shape = [num_unrollings * batch_size]
                new_inp_shape = [num_unrollings, batch_size]
                lbls_for_loss_shape = [num_unrollings, batch_size, self._vec_dim]
            inputs = tf.reshape(inputs, new_inp_shape)
            labels = tf.reshape(labels, prepared_lbls_shape)
            inputs = tf.one_hot(inputs, self._vocabulary_size)
            labels = tf.one_hot(labels, self._vocabulary_size)
            self._hooks['labels_prepared'] = labels
            labels = tf.reshape(
                labels,
                shape=lbls_for_loss_shape)
            return inputs, labels

    def _distribute_by_gpus(self, inputs, labels):
        with tf.device(self._base_device):
            inputs = tf.split(inputs, self._batch_sizes_on_gpus, 1, name='inp_on_dev')
            inputs_by_device = list()
            for dev_idx, device_inputs in enumerate(inputs):
                inputs_by_device.append(device_inputs)

            labels = tf.split(labels, self._batch_sizes_on_gpus, 1)
            labels_by_device = list()
            for dev_idx, device_labels in enumerate(labels):
                labels_by_device.append(
                    tf.reshape(device_labels,
                               [-1, self._vec_dim],
                               name='labels_on_dev_%s' % dev_idx))
            return inputs_by_device, labels_by_device

    def __init__(
            self,
            batch_size=64,
            num_layers=2,
            num_nodes=None,
            num_output_layers=1,
            num_output_nodes=None,
            vocabulary_size=None,
            embedding_size=128,
            num_unrollings=10,
            init_parameter=3.,
            num_gpus=1,
            regularization_rate=.000006,
            additional_metrics=None,
            regime='autonomous_training',
            going_to_limit_memory=False,
            optimizer='adam'
    ):
        """4 regimes are possible: autonomous_training, inference, training_with_meta_optimizer, optimizer_training"""

        if num_nodes is None:
            num_nodes = [112, 113]
        if num_output_nodes is None:
            num_output_nodes = list()
        if additional_metrics is None:
            additional_metrics = list()

        self._batch_size = batch_size
        self._num_layers = num_layers
        self._num_nodes = num_nodes
        self._vocabulary_size = vocabulary_size
        self._embedding_size = embedding_size
        self._num_output_layers = num_output_layers
        self._num_output_nodes = num_output_nodes
        self._num_unrollings = num_unrollings
        self._init_parameter = init_parameter
        self._regularization_rate = regularization_rate
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
            validation_labels_prepared=None,
            validation_predictions=None,
            reset_validation_state=None,
            reset_pupil_train_state=None,
            randomize_sample_state=None,
            dropout=None,
            saver=None)
        for add_metric in self._additional_metrics:
            self._hooks[add_metric] = None
            self._hooks['validation_' + add_metric] = None

        if not going_to_limit_memory:
            gpu_names = get_available_gpus()
            self._gpu_names = ['/gpu:%s' % i for i in range(len(gpu_names))]
        else:
            self._gpu_names = ['/gpu:%s' % i for i in range(num_gpus)]
        num_available_gpus = len(self._gpu_names)
        num_gpus, self._batch_sizes_on_gpus = get_num_gpus_and_bs_on_gpus(
            self._batch_size, num_gpus, num_available_gpus)
        self._num_gpus = num_gpus

        self._vec_dim = self._vocabulary_size

        if self._num_gpus == 1:
            self._base_device = '/gpu:0'
        else:
            self._base_device = '/cpu:0'

        self._dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self._hooks['dropout'] = self._dropout_keep_prob

        self._applicable_trainable = dict()

        self._train_storage = dict()
        self._inference_storage = dict()

        self._train_inputs_and_labels_placeholders = dict()     
        self._autonomous_train_specific_placeholders = dict()
        self._inference_placeholders = dict()

        if regime == 'autonomous_training':
            self._add_trainable_variables()
            self._add_train_storage()
            self._add_autonomous_training_specific_placeholders()
            self._add_train_inputs_and_labels_placeholders()
            
            self._train_graph()
            self._validation_graph()

        elif regime == 'inference':
            self._add_trainable_variables()

            self._validation_graph()

        elif regime == 'training_with_meta_optimizer':
            self._add_trainable_variables()
            self._add_train_storage()
            self._add_train_inputs_and_labels_placeholders()

            self._validation_graph()

        elif regime == 'optimizer_training':
            self._add_trainable_variables()
            self._add_train_storage()
            self._add_train_inputs_and_labels_placeholders()

            self._validation_graph()

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
            vocabulary_size=self._vocabulary_size,
            embedding_size=self._embedding_size,
            num_nodes=self._num_nodes,
            num_output_nodes=self._num_output_nodes + [self._vocabulary_size],
            num_layers=self._num_layers,
            num_output_layers=self._num_output_layers,
            batch_size=self._batch_size,
            num_unrollings=self._num_unrollings
        )

    def get_layer_dims(self):
        dims = dict()
        dims['embedding_layer'] = (self._vocabulary_size, self._embedding_size)
        dims['lstm_layers'] = []
        for layer_idx, _ in enumerate(self._num_nodes):
            in_dim, out_dim, _ = self._compute_lstm_matrix_parameters(layer_idx)
            dims['lstm_layers'].append((in_dim, out_dim))
        dims['output_layers'] = []
        for layer_idx, n_nodes in enumerate(self._num_output_nodes + [self._vocabulary_size]):
            in_dim, out_dim, _ = self._compute_output_matrix_parameters(layer_idx)
            dims['output_layers'].append((in_dim, out_dim))
        return dims
