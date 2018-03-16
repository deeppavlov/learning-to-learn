from __future__ import print_function
import numpy as np
import tensorflow as tf
from useful_functions import (construct, create_vocabulary,
                              get_positions_in_vocabulary, char2vec, pred2vec, pred2vec_fast, vec2char,
                              vec2char_fast, char2id, id2char, flatten, get_available_gpus, device_name_scope,
                              average_gradients, get_num_gpus_and_bs_on_gpus)


url = 'http://mattmahoney.net/dc/'


class LstmBatchGenerator(object):

    @staticmethod
    def create_vocabulary(texts):
        text = ''
        for t in texts:
            text += t
        return create_vocabulary(text)

    @staticmethod
    def char2vec(char,character_positions_in_vocabulary, speaker_idx, speaker_flag_size):
        return np.reshape(char2vec(char,character_positions_in_vocabulary), (1, 1, -1))

    @staticmethod
    def pred2vec(pred, speaker_idx, speaker_flag_size, batch_gen_args):
        return np.reshape(pred2vec(pred), (1, 1, -1))

    @staticmethod
    def vec2char(vec, vocabulary):
        return vec2char(vec, vocabulary)

    @staticmethod
    def vec2char_fast(vec, vocabulary):
        return vec2char(vec, vocabulary)

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabulary = vocabulary
        self._vocabulary_size = len(self.vocabulary)
        self.character_positions_in_vocabulary = get_positions_in_vocabulary(self.vocabulary)
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()

    def get_dataset_length(self):
        return len(self._text)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    def _start_batch(self):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id('\n', self.character_positions_in_vocabulary)] = 1.0
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
    def char2vec(char,character_positions_in_vocabulary, speaker_idx, speaker_flag_size):
        return np.array([char2id(char,character_positions_in_vocabulary)])

    @staticmethod
    def pred2vec(pred, speaker_idx, speaker_flag_size, batch_gen_args):
        return np.reshape(pred2vec_fast(pred), (1, -1, 1))

    @staticmethod
    def vec2char(vec, vocabulary):
        return vec2char(vec, vocabulary)

    @staticmethod
    def vec2char_fast(vec, vocabulary):
        return vec2char_fast(vec, vocabulary)

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabulary = vocabulary
        self._vocabulary_size = len(self.vocabulary)
        self.character_positions_in_vocabulary = get_positions_in_vocabulary(self.vocabulary)
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()

    def get_dataset_length(self):
        return len(self._text)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    def _start_batch(self):
        return np.array([[char2id('\n', self.character_positions_in_vocabulary)] for _ in range(self._batch_size)])

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
        if self._number_of_punctuation_marks is None:
            return dict()
        else:
            return {'mark_vec_len': self._mark_vec_len}

    @staticmethod
    def form_kwargs(kwargs_for_building, insertions):
        for insertion in insertions:
            if insertion['list_index'] is None:
                kwargs_for_building[insertion['hp_name']] = insertion['paste']
            else:
                kwargs_for_building[insertion['hp_name']][insertion['list_index']] = insertion['paste']
        return kwargs_for_building

    def _lstm_layer(self, inp, state, layer_idx):
        with tf.name_scope('lstm_layer_%s' % layer_idx):
            matr = self._lstm_matrices[layer_idx]
            bias = self._lstm_biases[layer_idx]
            nn = self._num_nodes[layer_idx]
            x = tf.concat([tf.nn.dropout(inp, self.dropout_keep_prob), state[0]], 1, name='X')
            linear_res = tf.add(tf.matmul(x, matr, name='matmul'), bias, name='linear_res')
            [sigm_arg, tanh_arg] = tf.split(linear_res, [3*nn, nn], axis=1, name='split_to_act_func_args')
            sigm_res = tf.sigmoid(sigm_arg, name='sigm_res')
            transform_vec = tf.tanh(tanh_arg, name='transformation_vector')
            [forget_gate, input_gate, output_gate] = tf.split(sigm_res, 3, axis=1, name='gates')
            new_cell_state = tf.add(forget_gate * state[1], input_gate * transform_vec, name='new_cell_state')
            new_hidden_state = tf.multiply(output_gate, tf.tanh(new_cell_state), name='new_hidden_state')
        return new_hidden_state, (new_hidden_state, new_cell_state)

    def _rnn_iter(self, embedding, all_states):
        with tf.name_scope('rnn_iter'):
            new_all_states = list()
            output = embedding
            for layer_idx, state in enumerate(all_states):
                output, state = self._lstm_layer(output, state, layer_idx)
                new_all_states.append(state)
            return output, new_all_states

    def _rnn_module(self, embeddings, all_states):
        rnn_outputs = list()
        with tf.name_scope('rnn_module'):
            for emb in embeddings:
                rnn_output, all_states = self._rnn_iter(emb, all_states)
                #print('rnn_output.shape:', rnn_output.get_shape().as_list())
                rnn_outputs.append(rnn_output)
        return rnn_outputs, all_states

    def _embed(self, inputs):
        with tf.name_scope('embeddings'):
            num_unrollings = len(inputs)
            inputs = tf.concat(inputs, 0, name='concatenated_inputs')
            embeddings = tf.matmul(inputs, self._embedding_matrix, name='embeddings_stacked')
            return tf.split(embeddings, num_unrollings, 0, name='embeddings')

    def _output_module(self, rnn_outputs):
        with tf.name_scope('output_module'):
            #print('rnn_outputs:', rnn_outputs)
            rnn_outputs = tf.concat(rnn_outputs, 0, name='concatenated_rnn_outputs')
            hs = rnn_outputs
            for layer_idx in range(self._num_output_layers):
                #print('hs.shape:', hs.get_shape().as_list())
                hs = tf.add(
                    tf.matmul(hs,
                              self._output_matrices[layer_idx],
                              name='matmul_in_%s_output_layer' % layer_idx),
                    self._output_biases[layer_idx],
                    name='res_of_%s_output_layer' % layer_idx)
                if layer_idx < self._num_output_layers - 1:
                    hs = tf.nn.relu(hs)
        return hs

    @staticmethod
    def _extract_op_name(full_name):
        scopes_stripped = full_name.split('/')[-1]
        return scopes_stripped.split(':')[0]

    def _compose_save_list(self,
                           *pairs):
        #print('start')
        with tf.name_scope('save_list'):
            save_list = list()
            for pair in pairs:
                #print('pair:', pair)
                variables = flatten(pair[0])
                #print(variables)
                new_values = flatten(pair[1])
                for variable, value in zip(variables, new_values):
                    name = self._extract_op_name(variable.name)
                    save_list.append(tf.assign(variable, value, name='assign_save_%s' % name))
            return save_list

    def _compose_reset_list(self, *args):
        with tf.name_scope('reset_list'):
            reset_list = list()
            flattened = flatten(args)
            for variable in flattened:
                shape = variable.get_shape().as_list()
                name = self._extract_op_name(variable.name)
                reset_list.append(tf.assign(variable, tf.zeros(shape), name='assign_reset_%s' % name))
            return reset_list

    def _compose_randomize_list(self, *args):
        with tf.name_scope('randomize_list'):
            randomize_list = list()
            flattened = flatten(args)
            for variable in flattened:
                shape = variable.get_shape().as_list()
                name = self._extract_op_name(variable.name)
                assign_tensor = tf.truncated_normal(shape, stddev=1.)
                #assign_tensor = tf.Print(assign_tensor, [assign_tensor], message='assign tensor:')
                assign = tf.assign(variable, assign_tensor, name='assign_reset_%s' % name)
                randomize_list.append(assign)
            return randomize_list

    def _compute_lstm_matrix_parameters(self, idx):
        if idx == 0:
            print(self._num_nodes)
            input_dim = self._num_nodes[0] + self._embedding_size
        else:
            input_dim = self._num_nodes[idx-1] + self._num_nodes[idx]
        output_dim = 4 * self._num_nodes[idx]
        stddev = self._init_parameter * np.sqrt(1./input_dim)
        return input_dim, output_dim, stddev

    def _compute_output_matrix_parameters(self, idx):
        if idx == 0:
            #print('self._num_nodes:', self._num_nodes)
            input_dim = self._num_nodes[-1]
        else:
            input_dim = self._num_output_nodes[idx-1]
        if idx == self._num_output_layers - 1:
            output_dim = self._vec_dim
        else:
            output_dim = self._num_output_nodes[idx]
        stddev = self._init_parameter * np.sqrt(1. / input_dim)
        return input_dim, output_dim, stddev

    def _l2_loss(self, matrices):
        with tf.name_scope('l2_loss'):
            regularizer = tf.contrib.layers.l2_regularizer(1.)
            loss = 0
            for matr in matrices:
                loss += regularizer(matr)
            return loss * self._regularization_rate

    def _train_graph(self):
        tower_grads = list()
        preds = list()
        losses = list()
        with tf.name_scope('train'):
            for gpu_batch_size, gpu_name, device_inputs, device_labels in zip(
                    self._batch_sizes_on_gpus, self._gpu_names, self._inputs_by_device, self._labels_by_device):
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

                        all_states = saved_states
                        embeddings = self._embed(device_inputs)
                        rnn_outputs, all_states = self._rnn_module(embeddings, all_states)
                        logits = self._output_module(rnn_outputs)

                        save_ops = self._compose_save_list((saved_states, all_states))

                        with tf.control_dependencies(save_ops):
                            all_matrices = [self._embedding_matrix]
                            all_matrices.extend(self._lstm_matrices)
                            all_matrices.extend(self._output_matrices)
                            l2_loss = self._l2_loss(all_matrices)

                            if self._number_of_punctuation_marks == 0:
                                loss = tf.reduce_mean(
                                    tf.nn.softmax_cross_entropy_with_logits(labels=device_labels, logits=logits))
                                concat_pred = tf.nn.softmax(logits)
                                preds.append(tf.split(concat_pred, self._num_unrollings))
                            else:
                                word_labels, punctuation_labels = tf.split(
                                    device_labels, [self._vocabulary_size, self._mark_vec_len],
                                    axis=1, name='word_and_punctuation_labels')
                                word_logits, punctuation_logits = tf.split(
                                    logits, [self._vocabulary_size, self._mark_vec_len],
                                    axis=1, name='word_and_punctuation_logits')

                                word_loss = tf.reduce_mean(
                                    tf.nn.softmax_cross_entropy_with_logits(
                                        labels=word_labels, logits=word_logits))
                                word_pred_concat = tf.nn.softmax(word_logits)

                                if self._punctuation_encoding == 'positional_notation':
                                    punctuation_pred_concat = tf.tanh(punctuation_logits)
                                    punctuation_loss = tf.divide(
                                        tf.reduce_sum(
                                            tf.nn.l2_loss(punctuation_labels - punctuation_logits)),
                                        self._batch_size,
                                        name='punctuation_loss')
                                elif self._punctuation_encoding == 'one_hot':
                                    separate_mark_labels = tf.split(
                                        punctuation_labels, self._max_mark_num, axis=1, name='separate_mark_labels')
                                    separate_mark_labels = tf.concat(
                                        separate_mark_labels, 0, name='separate_mark_labels_concat')
                                    separate_mark_logits = tf.split(
                                        punctuation_logits, self._max_mark_num, axis=1, name='separate_mark_logits')
                                    separate_mark_logits = tf.concat(
                                        separate_mark_logits, 0, name='separate_mark_logits_concat')

                                    punctuation_loss = tf.reduce_mean(
                                        tf.nn.softmax_cross_entropy_with_logits(
                                            labels=separate_mark_labels, logits=separate_mark_logits))

                                    separate_mark_preds = tf.nn.softmax(
                                        separate_mark_logits, name='separate_mark_preds')
                                    separate_mark_preds = tf.split(separate_mark_preds, self._max_mark_num, axis=0)
                                    punctuation_pred_concat = tf.concat(separate_mark_preds, 1)

                                concat_pred = tf.concat([word_pred_concat, punctuation_pred_concat], 1)
                                preds.append(tf.split(concat_pred, self._num_unrollings))

                                loss = word_loss + punctuation_loss
                            losses.append(loss)
                            # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                            grads_and_vars = optimizer.compute_gradients(loss + l2_loss)
                            tower_grads.append(grads_and_vars)

                            # splitting concatenated results for different characters
            with tf.device(self._base_device):
                with tf.name_scope(device_name_scope(self._base_device) + '_gradients'):
                    # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    grads_and_vars = average_gradients(tower_grads)
                    grads, v = zip(*grads_and_vars)
                    grads, _ = tf.clip_by_global_norm(grads, 1.)
                    self.train_op = optimizer.apply_gradients(zip(grads, v))
                    self._hooks['train_op'] = self.train_op

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
                    l = 0
                    for loss, gpu_batch_size in zip(losses, self._batch_sizes_on_gpus):
                        l += float(gpu_batch_size) / float(self._batch_size) * loss
                    self.loss = l
                    self._hooks['loss'] = self.loss

    def _validation_graph(self):
        with tf.device(self._gpu_names[0]):
            with tf.name_scope('validation'):
                if self._embeddings_in_batch:
                    self.sample_input = tf.placeholder(tf.float32,
                                                       shape=[1, 1, self._vec_dim],
                                                       name='sample_input')
                    self.validation_labels = tf.placeholder(tf.float32)
                    validation_labels_prepared = self.validation_labels
                    sample_input = self.sample_input
                else:
                    self.validation_labels = tf.placeholder(tf.int32, [1, self._max_mark_num + 1])
                    self.sample_input = tf.placeholder(tf.int32,
                                                       shape=[1, 1, self._max_mark_num + 1],
                                                       name='sample_input')
                    if self._max_mark_num > 0:
                        if self._punctuation_encoding == 'one_hot':
                            inputs = tf.unstack(self.sample_input, axis=2)
                            inp0 = tf.one_hot(inputs[0], self._vocabulary_size)
                            inps = [tf.one_hot(inp, self._number_of_punctuation_marks) for inp in inputs[1:]]
                            sample_input = tf.concat([inp0] + inps, 2)
                            labels = tf.unstack(self.validation_labels, axis=1)
                            lbl0 = tf.one_hot(labels[0], self._vocabulary_size)
                            lbls = [tf.one_hot(lbl, self._number_of_punctuation_marks) for lbl in labels[1:]]
                            validation_labels_prepared = tf.concat([lbl0] + lbls, 1)
                    else:
                        inputs = tf.reshape(self.sample_input, [1, -1])
                        sample_input = tf.one_hot(inputs, self._vocabulary_size)
                        labels = tf.reshape(self.validation_labels, [1])
                        validation_labels_prepared = tf.one_hot(labels, self._vocabulary_size)

                sample_input = tf.reshape(sample_input, [1, -1])

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

                reset_list = self._compose_reset_list(saved_sample_state)

                self.reset_sample_state = tf.group(*reset_list)
                self._hooks['reset_validation_state'] = self.reset_sample_state

                randomize_list = self._compose_randomize_list(saved_sample_state)

                self.randomize = tf.group(*randomize_list)
                self._hooks['randomize_sample_state'] = self.randomize

                embeddings = self._embed([sample_input])
                # print('embeddings:', embeddings)
                rnn_output, sample_state = self._rnn_module(embeddings, saved_sample_state)
                sample_logit = self._output_module(rnn_output)

                sample_save_ops = self._compose_save_list((saved_sample_state, sample_state))

                with tf.control_dependencies(sample_save_ops):
                    if self._number_of_punctuation_marks == 0:
                        self.sample_prediction = tf.nn.softmax(sample_logit)
                    else:
                        word_logit, punctuation_logit = tf.split(
                            sample_logit, [self._vocabulary_size, self._mark_vec_len], axis=1)
                        word_pred = tf.nn.softmax(word_logit)
                        if self._punctuation_encoding == 'positional_notation':
                            punctuation_pred = tf.tanh(punctuation_logit)

                        elif self._punctuation_encoding == 'one_hot':
                            separate_mark_logits = tf.split(
                                punctuation_logit, self._max_mark_num, axis=1, name='separate_mark_logits')
                            separate_mark_logits = tf.concat(
                                separate_mark_logits, 0, name='separate_mark_logits_concat')
                            separate_mark_preds = tf.nn.softmax(
                                separate_mark_logits, name='separate_mark_preds')
                            separate_mark_preds = tf.split(separate_mark_preds, self._max_mark_num, axis=0)
                            punctuation_pred = tf.concat(separate_mark_preds, 1)
                        self.sample_prediction = tf.concat([word_pred, punctuation_pred], 1)
                    self._hooks['validation_predictions'] = self.sample_prediction

    def __init__(self,
                 batch_size=64,
                 embeddings_in_batch=True,
                 num_layers=2,
                 num_nodes=[112, 113],
                 num_output_layers=1,
                 num_output_nodes=[],
                 vocabulary_size=None,
                 embedding_size=128,
                 num_unrollings=10,
                 init_parameter=3.,
                 num_gpus=1,
                 regularization_rate=.000006,
                 regime='train',
                 going_to_limit_memory=False,
                 number_of_punctuation_marks=0,
                 max_mark_num=0,
                 punctuation_encoding='one_hot'):

        self._hooks = dict(inputs=None,
                           labels=None,
                           labels_prepared=None,
                           train_op=None,
                           learning_rate=None,
                           loss=None,
                           predictions=None,
                           validation_inputs=None,
                           validation_labels=None,
                           validation_labels_prepared=None,
                           validation_predictions=None,
                           reset_validation_state=None,
                           randomize_sample_state=None,
                           dropout=None,
                           saver=None)

        self._batch_size = batch_size
        self._embeddings_in_batch = embeddings_in_batch
        self._num_layers = num_layers
        self._num_nodes = num_nodes
        self._vocabulary_size = vocabulary_size
        self._embedding_size = embedding_size
        self._num_output_layers = num_output_layers
        self._num_output_nodes = num_output_nodes
        self._num_unrollings = num_unrollings
        self._init_parameter = init_parameter
        self._regularization_rate = regularization_rate

        if not going_to_limit_memory:
            gpu_names = get_available_gpus()
            self._gpu_names = ['/gpu:%s' % i for i in range(len(gpu_names))]
        else:
            self._gpu_names = ['/gpu:%s' % i for i in range(num_gpus)]
        num_available_gpus = len(self._gpu_names)
        num_gpus, self._batch_sizes_on_gpus = get_num_gpus_and_bs_on_gpus(
            self._batch_size, num_gpus, num_available_gpus)
        self._num_gpus = num_gpus
        if number_of_punctuation_marks == 0:
            self._number_of_punctuation_marks = 0
        else:
            self._number_of_punctuation_marks = number_of_punctuation_marks + 1
        self._punctuation_encoding = punctuation_encoding
        self._max_mark_num = max_mark_num
        if self._number_of_punctuation_marks > 0:
            if self._punctuation_encoding == 'positional_notation':
                one_mark_bits = round(np.log2(self._number_of_punctuation_marks))
                self._mark_vec_len = self._max_mark_num * one_mark_bits
            elif self._punctuation_encoding == 'one_hot':
                one_mark_bits = self._number_of_punctuation_marks
                self._mark_vec_len = self._max_mark_num * one_mark_bits
            self._vec_dim = self._vocabulary_size + self._mark_vec_len
        else:
            self._vec_dim = self._vocabulary_size
            self._mark_vec_len = None

        if self._num_gpus == 1:
            self._base_device = '/gpu:0'
        else:
            self._base_device = '/cpu:0'
        with tf.device(self._base_device):
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

            if self._embeddings_in_batch:
                # print([self._num_unrollings, self._batch_size, self._vec_dim])
                self.inputs = tf.placeholder(
                    tf.float32, shape=[self._num_unrollings, self._batch_size, self._vec_dim])
                self.labels = tf.placeholder(
                    tf.float32, shape=[self._num_unrollings * self._batch_size, self._vec_dim])
                inputs = self.inputs
                labels = self.labels
            else:
                self.inputs = tf.placeholder(
                    tf.int32, shape=[self._num_unrollings, self._batch_size, self._max_mark_num + 1])
                self.labels = tf.placeholder(
                    tf.int32, shape=[self._num_unrollings * self._batch_size, self._max_mark_num + 1])
                if self._max_mark_num > 0:
                    if punctuation_encoding == 'one_hot':
                        inputs = tf.unstack(self.inputs, self._max_mark_num + 1, axis=2)
                        labels = tf.unstack(self.labels, self._max_mark_num + 1, axis=1)
                        inp0 = tf.one_hot(inputs[0], self._vocabulary_size)
                        other_inps = [tf.one_hot(inp, self._number_of_punctuation_marks) for inp in inputs[1:]]
                        lbl0 = tf.one_hot(labels[0], self._vocabulary_size)
                        other_lbls = [tf.one_hot(lbl, self._number_of_punctuation_marks) for lbl in labels[1:]]
                        inputs = tf.concat([inp0] + other_inps, 2)
                        labels = tf.concat([lbl0] + other_lbls, 1)
                else:
                    inputs = tf.reshape(self.inputs, [self._num_unrollings, self._batch_size])
                    labels = tf.reshape(self.labels, [self._num_unrollings * self._batch_size])
                    inputs = tf.one_hot(inputs, self._vocabulary_size)
                    labels = tf.one_hot(labels, self._vocabulary_size)



            #in_flags
            self._hooks['dropout'] = self.dropout_keep_prob
            self._hooks['inputs'] = self.inputs
            self._hooks['labels'] = self.labels
            self._hooks['labels_prepared'] = labels

            # inputs = tf.Print(inputs, [tf.argmax(inputs, axis=2)], message='inputs:', summarize=1200)
            # labels = tf.Print(labels, [tf.argmax(labels, axis=1)], message='labels_in_lstm:', summarize=1200)

            inputs = tf.split(inputs, self._batch_sizes_on_gpus, 1)
            self._inputs_by_device = list()
            for dev_idx, device_inputs in enumerate(inputs):
                self._inputs_by_device.append(tf.unstack(device_inputs, name='inp_on_dev_%s' % dev_idx))

            labels = tf.reshape(
                labels,
                shape=(self._num_unrollings, self._batch_size, self._vec_dim))

            labels = tf.split(labels, self._batch_sizes_on_gpus, 1)
            self._labels_by_device = list()
            for dev_idx, device_labels in enumerate(labels):
                self._labels_by_device.append(
                    tf.reshape(device_labels,
                               [-1, self._vec_dim],
                               name='labels_on_dev_%s' % dev_idx))

            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self._hooks['learning_rate'] = self.learning_rate

            self._embedding_matrix = tf.Variable(
                tf.truncated_normal([self._vec_dim, self._embedding_size],
                                    stddev=self._init_parameter*np.sqrt(1./self._vec_dim)),
                name='embedding_matrix')

            self._lstm_matrices = list()
            self._lstm_biases = list()
            for layer_idx in range(self._num_layers):
                input_dim, output_dim, stddev = self._compute_lstm_matrix_parameters(layer_idx)
                self._lstm_matrices.append(tf.Variable(tf.truncated_normal([input_dim,
                                                                            output_dim],
                                                                           stddev=stddev),
                                                       name='lstm_matrix_%s' % layer_idx))
                self._lstm_biases.append(tf.Variable(tf.zeros([output_dim]), name='lstm_bias_%s' % layer_idx))
            self._output_matrices = list()
            self._output_biases = list()
            for layer_idx in range(self._num_output_layers):
                input_dim, output_dim, stddev = self._compute_output_matrix_parameters(layer_idx)
                #print('input_dim:', input_dim)
                #print('output_dim:', output_dim)
                self._output_matrices.append(tf.Variable(tf.truncated_normal([input_dim, output_dim],
                                                                             stddev=stddev),
                                                         name='output_matrix_%s' % layer_idx))
                self._output_biases.append(tf.Variable(
                    tf.zeros([output_dim]),
                    name='output_bias_%s' % layer_idx))
        with tf.device('/cpu:0'):
            saved_vars = dict()
            saved_vars['embedding_matrix'] = self._embedding_matrix
            for layer_idx, lstm_matrix in enumerate(self._lstm_matrices):
                saved_vars['lstm_matrix_%s' % layer_idx] = lstm_matrix
                saved_vars['lstm_bias_%s' % layer_idx] = self._lstm_biases[layer_idx]
            for layer_idx, (output_matrix, output_bias) in enumerate(zip(self._output_matrices, self._output_biases)):
                saved_vars['output_matrix_%s' % layer_idx] = output_matrix
                saved_vars['output_bias_%s' % layer_idx] = output_bias
            self.saver = tf.train.Saver(saved_vars, max_to_keep=None)
            self._hooks['saver'] = self.saver

        if regime == 'train':
            self._train_graph()
            self._validation_graph()
        if regime == 'inference':
            self._validation_graph()

    def get_default_hooks(self):
        return dict(self._hooks.items())

    def get_building_parameters(self):
        pass


