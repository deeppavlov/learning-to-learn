from subword_nmt.apply_bpe import BPE
import numpy as np
from useful_functions import (char2vec, pred2vec, vec2char, get_positions_in_vocabulary,
                              char2id, id2char, pred2vec_fast, vec2char_fast)
import re

MAX_NUM_PUNCTUATION_MARKS = 6


def prepare_for_bpe(text):
    punctuation_marks = re.escape('!"\'(),-.:;?')
    others = re.escape('%=^~№/\n')

    regex = punctuation_marks + others

    text = re.sub('([%s])' % regex, r' \1 ', text)
    text = re.sub(' {2,}', ' ', text)
    return text


def bpe_post_processing(text):
    chars_not_preceded_by_space = re.escape('%^!"\'),.:;?')
    chars_not_followed_by_space = re.escape('№')
    chars_to_highlight = re.escape('\n')

    text = re.sub(' ([%s])' % chars_not_preceded_by_space, r'@@ \1', text)
    text = re.sub('([%s]) ' % chars_not_followed_by_space, r'\1@@ ', text)
    text = re.sub('([%s])' % chars_to_highlight, r'@@ \1@@ ', text)
    return text


def create_vocabulary(text):
    vocabulary = list()
    text = re.sub('@@', '', text)
    segments = text.split()
    for s in segments:
        if s not in vocabulary and len(s) > 0:
            vocabulary.append(s)
    vocabulary.append(' ')
    if '\n' not in vocabulary:
        vocabulary.append('\n')
    return sorted(vocabulary)


class BpeBatchGenerator(object):

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
        return np.reshape(pred2vec(pred), (1, -1, 1))

    @staticmethod
    def vec2char(vec, vocabulary):
        return vec2char(vec, vocabulary)

    @staticmethod
    def vec2char_fast(vec, vocabulary):
        return vec2char(vec, vocabulary)

    @staticmethod
    def make_pairs(text, batch_gen_args):
        splitted = [s for s in re.split('@@ ?', text)]
        pairs = list()
        for s in splitted:
            sp = re.split('( )', s)
            for sp_ in sp:
                if len(sp_) > 0:
                    pairs.append(sp_)
        # print(pairs[:1000])
        return pairs

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):
        self._text = text
        self._pairs = self.make_pairs(self._text, None)
        self._number_of_pairs = len(self._pairs)
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabulary = vocabulary
        self._vocabulary_size = len(self.vocabulary)
        self.character_positions_in_vocabulary = get_positions_in_vocabulary(self.vocabulary)
        self._num_unrollings = num_unrollings
        segment = self._number_of_pairs // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()

    def get_dataset_length(self):
        return len(self._pairs)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    def _start_batch(self):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id('\n', self.character_positions_in_vocabulary)] = 1.0
        return batch

    def _zero_batch(self):
        return -np.ones(shape=(self._batch_size), dtype=np.float)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            # print('len(self._pairs):', len(self._pairs))
            # print('self._cursor[b]:', self._cursor[b])
            batch[b, char2id(self._pairs[self._cursor[b]], self.character_positions_in_vocabulary)] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._number_of_pairs
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
        # print('(BpeBatchGenerator.next)batches:', batches)
        self._last_batch = batches[-1]
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0)

    def _next_batch_with_tokens(self):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        tokens = list()
        for b in range(self._batch_size):
            # print('len(self._pairs):', len(self._pairs))
            # print('self._cursor[b]:', self._cursor[b])
            tokens.append(self._pairs[self._cursor[b]])
            batch[b, char2id(self._pairs[self._cursor[b]], self.character_positions_in_vocabulary)] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._number_of_pairs
        return batch, tokens

    def next_with_tokens(self):
        batches = [self._last_batch]
        batch, tokens = self._next_batch_with_tokens()
        batches.append(batch)
        self._last_batch = batches[-1]
        # print('(BpeBatchGenerator.next_with_tokens)tokens:', tokens)
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0), tokens


class BpeFastBatchGenerator(object):

    @staticmethod
    def create_vocabulary(texts):
        text = ''
        for t in texts:
            text += t
        return create_vocabulary(text)

    @staticmethod
    def char2vec(char, characters_positions_in_vocabulary, speaker_idx, speaker_flag_size):
        return np.reshape(np.array([char2id(char, characters_positions_in_vocabulary)]), (1, 1, 1))

    @staticmethod
    def pred2vec(pred, speaker_idx, speaker_flag_size, batch_gen_args):
        return np.reshape(pred2vec_fast(pred), (1, -1, 1))

    @staticmethod
    def vec2char(vec, vocabulary):
        return vec2char(vec, vocabulary)

    @staticmethod
    def vec2char_fast(vec, vocabulary):
        return vec2char_fast(vec, vocabulary)

    @staticmethod
    def make_pairs(text, batch_gen_args):
        splitted = [s for s in re.split('@@ ?', text)]
        pairs = list()
        for s in splitted:
            sp = re.split('( )', s)
            for sp_ in sp:
                if len(sp_) > 0:
                    pairs.append(sp_)
        # print(pairs[:1000])
        return pairs

    @staticmethod
    def _create_id_array(pairs, character_positions_in_vocabulary):
        number_of_pairs = len(pairs)
        ids = np.ndarray(shape=(number_of_pairs), dtype=np.int16)
        for p_idx, p in enumerate(pairs):
            ids[p_idx] = char2id(p, character_positions_in_vocabulary)
        return ids

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):
        self._text = text
        self._pairs = self.make_pairs(self._text, None)
        self._number_of_pairs = len(self._pairs)
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabulary = vocabulary
        self._vocabulary_size = len(self.vocabulary)
        self.character_positions_in_vocabulary = get_positions_in_vocabulary(self.vocabulary)
        self._ids = self._create_id_array(self._pairs, self.character_positions_in_vocabulary)
        self._num_unrollings = num_unrollings
        segment = self._number_of_pairs // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()
        # print('(BpeFastBatchGenerator.__init__)len(self._pairs):', len(self._pairs))

    def get_dataset_length(self):
        return len(self._pairs)

    def get_vocabulary_size(self):
        return self._vocabulary_size

    def _start_batch(self):
        return np.array([[char2id('\n', self.character_positions_in_vocabulary)] for _ in range(self._batch_size)])

    def _zero_batch(self):
        return -np.ones(shape=(self._batch_size), dtype=np.float)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        ret = np.array([[self._ids[self._cursor[b]]]
                        for b in range(self._batch_size)])
        # print('(BpeFastBatchGenerator._next_batch)pairs:',
        #       [self._pairs[self._cursor[b]] for b in range(self._batch_size)])
        for b in range(self._batch_size):
            self._cursor[b] = (self._cursor[b] + 1) % self._number_of_pairs
        return ret

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0)

    def _next_batch_with_tokens(self):
        tokens = list()
        bs = list()
        for b in range(self._batch_size):
            # print('len(self._pairs):', len(self._pairs))
            # print('self._cursor[b]:', self._cursor[b])
            tokens.append(self._pairs[self._cursor[b]])
            bs.append(np.array([char2id(self._pairs[self._cursor[b]], self.character_positions_in_vocabulary)]))
            self._cursor[b] = (self._cursor[b] + 1) % self._number_of_pairs
        return np.stack(bs), tokens

    def next_with_tokens(self):
        batches = [self._last_batch]
        batch, tokens = self._next_batch_with_tokens()
        batches.append(batch)
        self._last_batch = batches[-1]
        # print('(BpeBatchGenerator.next_with_tokens)tokens:', tokens)
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0), tokens


def create_vocabularies_one_hot(text, punctuation_marks):
    vocabulary = list()
    text = re.sub('@@', '', text)
    segments = text.split()
    for s in segments:
        if s not in vocabulary and len(s) > 0 and s not in punctuation_marks:
            vocabulary.append(s)
    vocabulary.append(' ')
    if '\n' not in vocabulary:
        vocabulary.append('\n')
    return sorted(vocabulary), sorted(punctuation_marks)


def char2vec_one_hot(pairs, character_positions_in_vocabulary):
    if not isinstance(pairs[0], tuple):
        pairs = [pairs]
    b_size = len(pairs)
    num_punc_marks = [len(pair) - 1 for pair in pairs]
    word_char_positions = character_positions_in_vocabulary[0]
    punctuation_char_positions = character_positions_in_vocabulary[1]
    word_voc_size = len(word_char_positions)
    punctuation_voc_size = len(punctuation_char_positions) + 1

    word_vec = np.zeros(shape=(b_size, word_voc_size), dtype=np.float)
    punc_vecs = [np.zeros(shape=(b_size, punctuation_voc_size), dtype=np.float)
                 for _ in range(MAX_NUM_PUNCTUATION_MARKS)]

    for b, pair in enumerate(pairs):
        word_vec[b, char2id(pair[0], word_char_positions)] = 1.0
        for punc_idx, punc_vec in enumerate(punc_vecs):
            if punc_idx < num_punc_marks[b]:
                # print('pair:', pair)
                # print('punc_idx:', punc_idx)
                # print('num_punc_marks[b]:', num_punc_marks[b])
                punc_vec[b, char2id(pair[punc_idx + 1], punctuation_char_positions) + 1] = 1.0
            else:
                punc_vec[b, 0] = 1.0
    np.set_printoptions(threshold=np.nan, linewidth=52)
    # print('(char2vec_one_hot)pairs:', pairs)
    # print('(char2vec_one_hot) returned:\n', np.reshape(np.concatenate(tuple([word_vec] + punc_vecs), axis=1), [-1]))
    return np.concatenate(tuple([word_vec] + punc_vecs), axis=1)


def split_into_word_and_punctuation(preds, punctuation_voc_size):
    shape1 = preds.shape[1]
    # print('(split_into_word_and_punctuation)shape1:', shape1)
    word_voc_size = shape1 - (punctuation_voc_size + 1) * MAX_NUM_PUNCTUATION_MARKS
    # print('(split_into_word_and_punctuation)word_voc_size:', word_voc_size)
    split_dims = [word_voc_size] + \
                 [word_voc_size + (punctuation_voc_size + 1) * i for i in range(1, MAX_NUM_PUNCTUATION_MARKS)]
    # print('(split_into_word_and_punctuation)split_dims:', split_dims)
    return np.split(preds, split_dims, axis=1)


def pred2vec_one_hot(preds, batch_gen_args):
    preds = split_into_word_and_punctuation(preds, batch_gen_args['punctuation_voc_size'])
    try:
        vec = [pred2vec(pred) for pred in preds]
    except ValueError:
        print('preds:', preds)
        raise
    return np.concatenate(tuple(vec), axis=1)


def vec2char_one_hot(vec, vocabularies):
    vecs = split_into_word_and_punctuation(vec, len(vocabularies[1]))
    chars = list()
    chars.append(vec2char(vecs[0], vocabularies[0]))
    extended_punc_voc = [''] + vocabularies[1]
    for vec in vecs[1:]:
        chars.append(vec2char(vec, extended_punc_voc))

    str_list = list()
    if isinstance(chars[0], list):
        for one_repl_chars in zip(*chars):
            str_list.append(''.join(one_repl_chars))
        return str_list
    return ''.join(chars)


class BpeBatchGeneratorOneHot(object):

    @staticmethod
    def create_vocabularies(texts, punctuation_marks):
        text = ''
        for t in texts:
            text += t
        return create_vocabularies_one_hot(text, punctuation_marks)

    @staticmethod
    def char2vec(char, character_positions_in_vocabulary, speaker_idx, speaker_flag_size):
        return np.reshape(char2vec_one_hot(char, character_positions_in_vocabulary), (1, 1, -1))

    @staticmethod
    def pred2vec(pred, speaker_idx, speaker_flag_size, batch_gen_args):
        return np.reshape(pred2vec_one_hot(pred, batch_gen_args), (1, 1, -1))

    @staticmethod
    def vec2char(vec, vocabularies):
        return vec2char_one_hot(vec, vocabularies)

    @staticmethod
    def vec2char_fast(vec, vocabulary):
        return vec2char_one_hot(vec, vocabulary)

    @staticmethod
    def make_pairs(text, batch_gen_args):
        punctuation_marks = batch_gen_args['punctuation_marks']
        splitted = [s for s in re.split('@@ ?', text)]
        all_tokens = list()
        for s in splitted:
            sp = re.split('( )', s)
            for sp_ in sp:
                if len(sp_) > 0:
                    all_tokens.append(sp_)
        # print('(BpeBatchGeneratorOneHot.make_pairs)text:', text)
        # print('(BpeBatchGeneratorOneHot.make_pairs)all_tokens:', all_tokens)
        pairs = list()
        pair = list()
        for t in all_tokens:
            if t not in punctuation_marks:
                if len(pair) > 0:
                    pairs.append(tuple(pair))
                pair = [t]
            else:
                if len(pair) > 0:
                    pair.append(t)
        if len(pair) > 0:
            pairs.append(tuple(pair))
        # print(pairs[:1000])
        return pairs

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabularies = vocabulary
        self._vocabulary_sizes = [len(voc) for voc in self.vocabularies]
        self.character_positions_in_vocabulary = [get_positions_in_vocabulary(voc) for voc in self.vocabularies]

        self._pairs = self.make_pairs(self._text, {'punctuation_marks': self.vocabularies[1]})
        self._number_of_pairs = len(self._pairs)

        self._num_unrollings = num_unrollings
        segment = self._number_of_pairs // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()

    def get_dataset_length(self):
        return len(self._pairs)

    def get_vocabulary_size(self):
        return self._vocabulary_sizes

    def _start_batch(self):
        word_batch = np.zeros(shape=(self._batch_size, self._vocabulary_sizes[0]), dtype=np.float)
        for b in range(self._batch_size):
            word_batch[b, char2id('\n', self.character_positions_in_vocabulary[0])] = 1.0
        no_punc_batch = np.zeros(
            shape=(self._batch_size, self._vocabulary_sizes[1] + 1), dtype=np.float)
        for b in range(self._batch_size):
            no_punc_batch[b, 0] = 1.0
        return np.concatenate(tuple([word_batch] + [no_punc_batch] * MAX_NUM_PUNCTUATION_MARKS), axis=1)

    def _zero_batch(self):
        return np.zeros(shape=(self._batch_size,
                               self._vocabulary_sizes[0] + MAX_NUM_PUNCTUATION_MARKS * self._vocabulary_sizes[1]),
                        dtype=np.float)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        pairs = [self._pairs[self._cursor[b]] for b in range(self._batch_size)]
        self._cursor = [(self._cursor[b] + 1) % self._number_of_pairs for b in range(self._batch_size)]
        return char2vec_one_hot(pairs, self.character_positions_in_vocabulary)

    def char2batch(self, char):
        return np.stack(char2vec_one_hot(char, self.character_positions_in_vocabulary)), np.stack(self._zero_batch())

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        shapes = [s.shape for s in batches[:-1]]
        # print('(BpeBatchGeneratorOneHot.next)shapes:', shapes)
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0)

    def _next_batch_with_tokens(self):
        batch = np.zeros(shape=(self._batch_size, self._vocabulary_size), dtype=np.float)
        tokens = list()
        for b in range(self._batch_size):
            # print('len(self._pairs):', len(self._pairs))
            # print('self._cursor[b]:', self._cursor[b])
            tokens.append(self._pairs[self._cursor[b]])
            batch[b, char2id(self._pairs[self._cursor[b]], self.character_positions_in_vocabulary)] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._number_of_pairs
        return batch, tokens

    def next_with_tokens(self):
        batches = [self._last_batch]
        batch, tokens = self._next_batch_with_tokens()
        batches.append(batch)
        self._last_batch = batches[-1]
        # print('(BpeBatchGenerator.next_with_tokens)tokens:', tokens)
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0), tokens


def char2vec_one_hot_fast(pairs, character_positions_in_vocabulary):
    if not isinstance(pairs[0], tuple):
        pairs = [pairs]
    b_size = len(pairs)
    num_punc_marks = [len(pair) - 1 for pair in pairs]
    word_char_positions = character_positions_in_vocabulary[0]
    punctuation_char_positions = character_positions_in_vocabulary[1]

    word_vec = np.zeros(shape=(b_size, 1), dtype=np.int32)
    punc_vecs = [np.zeros(shape=(b_size, 1), dtype=np.int32)
                 for _ in range(MAX_NUM_PUNCTUATION_MARKS)]

    for b, pair in enumerate(pairs):
        word_vec[b, 0] = char2id(pair[0], word_char_positions)
        for punc_idx, punc_vec in enumerate(punc_vecs):
            if punc_idx < num_punc_marks[b]:
                # print('pair:', pair)
                # print('punc_idx:', punc_idx)
                # print('num_punc_marks[b]:', num_punc_marks[b])
                punc_vec[b, 0] = char2id(pair[punc_idx + 1], punctuation_char_positions) + 1
            else:
                punc_vec[b, 0] = 0
    np.set_printoptions(threshold=np.nan, linewidth=52)
    # print('(char2vec_one_hot)pairs:', pairs)
    # print('(char2vec_one_hot) returned:\n', np.reshape(np.concatenate(tuple([word_vec] + punc_vecs), axis=1), [-1]))
    return np.concatenate(tuple([word_vec] + punc_vecs), axis=1)


def split_into_word_and_punctuation_fast(preds):
    shape1 = preds.shape[1]
    # print('(split_into_word_and_punctuation)shape1:', shape1)
    return np.split(preds, shape1, axis=1)


def vec2char_one_hot_fast(vec, vocabularies):
    vecs = split_into_word_and_punctuation_fast(vec)
    chars = list()
    chars.append(vec2char_fast(vecs[0], vocabularies[0]))
    extended_punc_voc = [''] + vocabularies[1]
    for vec in vecs[1:]:
        chars.append(vec2char_fast(vec, extended_punc_voc))

    str_list = list()
    if isinstance(chars[0], list):
        for one_repl_chars in zip(*chars):
            str_list.append(''.join(one_repl_chars))
        return str_list
    return ''.join(chars)


def pred2vec_one_hot_fast(preds, batch_gen_args):
    preds = split_into_word_and_punctuation(preds, batch_gen_args['punctuation_voc_size'])
    try:
        vec = [pred2vec_fast(pred) for pred in preds]
    except ValueError:
        print('preds:', preds)
        raise
    print('(pred2vec_one_hot_fast)vec:', vec)
    return np.stack(tuple(vec), axis=1)


class BpeFastBatchGeneratorOneHot(object):

    @staticmethod
    def create_vocabularies(texts, punctuation_marks):
        text = ''
        for t in texts:
            text += t
        return create_vocabularies_one_hot(text, punctuation_marks)

    @staticmethod
    def char2vec(char, character_positions_in_vocabulary, speaker_idx, speaker_flag_size):
        return np.reshape(char2vec_one_hot_fast(char, character_positions_in_vocabulary), (1, 1, -1))

    # @staticmethod
    # def pred2vec(pred, speaker_idx, speaker_flag_size, batch_gen_args):
    #     return np.reshape(pred2vec_one_hot(pred, batch_gen_args), (1, 1, -1))

    @staticmethod
    def pred2vec(pred, speaker_idx, speaker_flag_size, batch_gen_args):
        return np.reshape(pred2vec_one_hot_fast(pred, batch_gen_args), (1, 1, -1))

    @staticmethod
    def vec2char(vec, vocabularies):
        return vec2char_one_hot(vec, vocabularies)

    @staticmethod
    def vec2char_fast(vec, vocabularies):
        return vec2char_one_hot_fast(vec, vocabularies)

    @staticmethod
    def make_pairs(text, batch_gen_args):
        punctuation_marks = batch_gen_args['punctuation_marks']
        splitted = [s for s in re.split('@@ ?', text)]
        all_tokens = list()
        for s in splitted:
            sp = re.split('( )', s)
            for sp_ in sp:
                if len(sp_) > 0:
                    all_tokens.append(sp_)
        # print('(BpeBatchGeneratorOneHot.make_pairs)text:', text)
        # print('(BpeBatchGeneratorOneHot.make_pairs)all_tokens:', all_tokens)
        pairs = list()
        pair = list()
        for t in all_tokens:
            if t not in punctuation_marks:
                if len(pair) > 0:
                    pairs.append(tuple(pair))
                pair = [t]
            else:
                if len(pair) > 0:
                    pair.append(t)
        if len(pair) > 0:
            pairs.append(tuple(pair))
        # print(pairs[:1000])
        return pairs

    @staticmethod
    def _create_id_array(pairs, character_positions_in_vocabulary):
        number_of_pairs = len(pairs)
        ids = np.zeros(shape=(number_of_pairs, MAX_NUM_PUNCTUATION_MARKS), dtype=np.int16)
        for p_idx, p in enumerate(pairs):
            for t_idx, (token, cpiv) in enumerate(zip(p, character_positions_in_vocabulary)):
                ids[p_idx, t_idx] = char2id(token, cpiv) + 1
        return ids

    def __init__(self, text, batch_size, num_unrollings=1, vocabulary=None):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocabularies = vocabulary
        self._vocabulary_sizes = [len(voc) for voc in self.vocabularies]
        self.character_positions_in_vocabulary = [get_positions_in_vocabulary(voc) for voc in self.vocabularies]
        self._pairs = self.make_pairs(self._text, {'punctuation_marks': self.vocabularies[1]})
        self._number_of_pairs = len(self._pairs)

        self._ids = self._create_id_array(self._pairs, self.character_positions_in_vocabulary)
        self._num_unrollings = num_unrollings
        segment = self._number_of_pairs // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._start_batch()

    def get_dataset_length(self):
        return len(self._pairs)

    def get_vocabulary_size(self):
        return self._vocabulary_sizes

    def _start_batch(self):
        return np.array([[char2id('\n', self.character_positions_in_vocabulary[0])] + [0] * MAX_NUM_PUNCTUATION_MARKS
                         for _ in range(self._batch_size)])

    def _zero_batch(self):
        return -np.ones(shape=(self._batch_size, MAX_NUM_PUNCTUATION_MARKS+1), dtype=np.float)

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        pairs = [self._pairs[self._cursor[b]] for b in range(self._batch_size)]
        self._cursor = [(self._cursor[b] + 1) % self._number_of_pairs for b in range(self._batch_size)]
        return char2vec_one_hot_fast(pairs, self.character_positions_in_vocabulary)

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        # shapes = [s.shape for s in batches[:-1]]
        # print('(BpeBatchGeneratorOneHot.next)shapes:', shapes)
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0)

    def _next_batch_with_tokens(self):
        tokens = list()
        bs = list()
        for b in range(self._batch_size):
            # print('len(self._pairs):', len(self._pairs))
            # print('self._cursor[b]:', self._cursor[b])
            tokens.append(self._pairs[self._cursor[b]])
            bs.append(self._pairs[self._cursor[b]])
            self._cursor[b] = (self._cursor[b] + 1) % self._number_of_pairs
        return char2vec_one_hot_fast(bs, self.character_positions_in_vocabulary), tokens

    def next_with_tokens(self):
        batches = [self._last_batch]
        batch, tokens = self._next_batch_with_tokens()
        batches.append(batch)
        self._last_batch = batches[-1]
        # print('(BpeBatchGenerator.next_with_tokens)tokens:', tokens)
        return np.stack(batches[:-1]), np.concatenate(batches[1:], 0), tokens
