import itertools
import importlib
import math
import re
import numpy as np
from functools import reduce
import os
import shutil
import ast
from collections import OrderedDict
import tensorflow as tf
from tensorflow.python.client import device_lib


escape_sequences = ['\\', '\'', '\"', '\a', '\b', '\f', '\n', '\r', '\t', '\v']
skipped_escape_sequences = ['\\', '\'', '\"']
escape_sequences_replacements = {'\\': '\\\\',
                                 '\'': '\\\'',
                                 '\"': '\\\"',
                                 '\a': '\\a',
                                 '\b': '\\b',
                                 '\f': '\\f',
                                 '\n': '\\n',
                                 '\r': '\\r',
                                 '\t': '\\t',
                                 '\v': '\\v'}
METRICS = ['accuracy', 'bpc', 'loss', 'perplexity']
DIGITS = list('0123456789')
NUMBER_CHARS = DIGITS + ['e', '+', '-', '.', ' ']
METRIC_IMPROVEMENT_DIRECTIONS = dict(
    accuracy='up',
    bpc='down',
    loss='down',
    perplexity='down',
)
EPSILON = 1e-10


class InvalidArgumentError(Exception):
    def __init__(self, msg, value, name, allowed_values):
        super(InvalidArgumentError, self).__init__(msg)
        self._msg = msg
        self._value = value
        self._name = name
        self._allowed_values = allowed_values


class WrongMethodCallError(Exception):
    def __init__(self, msg):
        super(WrongMethodCallError, self).__init__(msg)
        self._msg = msg


class MissingHPError(Exception):
    def __init__(self, hp_name, message):
        self.hp_name = hp_name
        self.message = message


class ExtraHPError(Exception):
    def __init__(self, hp_order, hp_names, message):
        self.hp_order = hp_order
        self.hp_names = hp_names
        self.message = message


class HeaderLineError(Exception):
    def __init__(self, header_line, message):
        self.header_line = header_line
        self.message = message


class BadFormattingError(Exception):
    def __init__(self, formatting, value, message):
        self.formatting = formatting
        self.value = value
        self.message = message


class FailedBroadcastError(Exception):
    def __init__(self, required_num, list_, message):
        self.required_num = required_num
        self.list_ = list_
        self.message = message


class DatasetSizeError(Exception):
    def __init__(self, provided_size, required_size, message):
        self.provided_size = provided_size
        self.required_size = required_size
        self.message = message


class HPLayoutMissingError(Exception):
    def __init__(self, eval_dir, message):
        self.eval_dir = eval_dir
        self.message = message


def create_vocabulary(text):
    all_characters = list()
    for char in text:
        if char not in all_characters:
            all_characters.append(char)
    return sorted(all_characters, key=lambda dot: ord(dot))


def get_positions_in_vocabulary(vocabulary):
    character_positions_in_vocabulary = dict()
    for idx, char in enumerate(vocabulary):
        character_positions_in_vocabulary[char] = idx
    return character_positions_in_vocabulary


def char2id(char, character_positions_in_vocabulary):
    if char in character_positions_in_vocabulary:
        return character_positions_in_vocabulary[char]
    else:
        print(u'Unexpected character: %s\nUnexpected character number: %s\n' %
              (repr(char), str([ord(c) for c in char])))
        raise ValueError


def id2char(dictid, vocabulary):
    voc_size = len(vocabulary)
    if (dictid >= 0) and (dictid < voc_size):
        return vocabulary[dictid]
    else:
        print(u"unexpected id")
        return u'\0'


def filter_text(text, allowed_letters):
    new_text = ""
    for char in text:
        if char in allowed_letters:
            new_text += char
    return new_text


def char2vec(char, character_positions_in_vocabulary):
    voc_size = len(character_positions_in_vocabulary)
    vec = np.zeros(shape=(1, voc_size), dtype=np.float)
    vec[0, char2id(char, character_positions_in_vocabulary)] = 1.0
    return vec


def pred2vec(pred):
    shape = pred.shape
    vecs = np.zeros(shape, dtype=np.float32)
    ids = np.argmax(pred, 1)
    for char_idx, char_id in enumerate(np.nditer(ids)):
        vecs[char_idx, char_id] = 1.
    return vecs


def pred2vec_fast(pred):
    ids = np.argmax(pred, 1)
    return ids


def vec2char(pred, vocabulary):
    char_list = list()
    ids = np.argmax(pred, 1)
    for id in np.nditer(ids):
        char_list.append(id2char(id, vocabulary))
    if len(char_list) > 1:
        return char_list[0]
    else:
        return char_list


def vec2char_fast(pred, vocabulary):
    char_list = list()
    for id in np.nditer(pred):
        char_list.append(id2char(id, vocabulary))
    if len(char_list) > 1:
        return char_list[0]
    else:
        return char_list


def char_2_base_vec(character_positions_in_vocabulary,
                    char):
    voc_size = len(character_positions_in_vocabulary)
    vec = np.zeros(shape=(1, voc_size), dtype=np.float32)
    vec[0, char2id(char, character_positions_in_vocabulary)] = 1.0
    return vec


def create_and_save_vocabulary(input_file_name,
                               vocabulary_file_name):
    input_f = open(input_file_name, 'r', encoding='utf-8')
    text = input_f.read()
    output_f = open(vocabulary_file_name, 'w', encoding='utf-8')
    vocabulary = create_vocabulary(text)
    vocabulary_string = ''.join(vocabulary)
    output_f.write(vocabulary_string)
    input_f.close()
    output_f.close()


def load_vocabulary_from_file(vocabulary_file_name):
    with open(vocabulary_file_name, 'r') as f:
        vocabulary_string = f.read()
    return list(vocabulary_string)


def check_not_one_byte(text):
    not_one_byte_counter = 0
    max_character_order_index = 0
    min_character_order_index = 2 ** 16
    present_characters = [0] * 256
    number_of_characters = 0
    for char in text:
        if ord(char) > 255:
            not_one_byte_counter += 1
        if len(present_characters) <= ord(char):
            present_characters.extend([0] * (ord(char) - len(present_characters) + 1))
            present_characters[ord(char)] = 1
            number_of_characters += 1
        elif present_characters[ord(char)] == 0:
            present_characters[ord(char)] = 1
            number_of_characters += 1
        if ord(char) > max_character_order_index:
            max_character_order_index = ord(char)
        if ord(char) < min_character_order_index:
            min_character_order_index = ord(char)
    return not_one_byte_counter, min_character_order_index, max_character_order_index, number_of_characters, present_characters


def construct(obj):
    """Used for preventing of not expected changing of class attributes"""
    if isinstance(obj, OrderedDict):
        new_obj = OrderedDict()
        for key, value in obj.items():
            new_obj[key] = construct(value)
    elif not isinstance(obj, OrderedDict) and isinstance(obj, dict):
        new_obj = dict()
        for key, value in obj.items():
            new_obj[key] = construct(value)
    elif isinstance(obj, list):
        new_obj = list()
        for value in obj:
            new_obj.append(construct(value))
    elif isinstance(obj, tuple):
        base = list()
        for value in obj:
            base.append(construct(value))
        new_obj = tuple(base)
    elif isinstance(obj, str):
        new_obj = str(obj)
    else:
        new_obj = obj
    # elif isinstance(obj, (int, float, complex, type(None))) or inspect.isclass(obj):
    #     new_obj = obj
    # else:
    #     raise TypeError("Object of unsupported type was passed to construct function: %s" % type(obj))
    return new_obj


def get_keys_from_nested(obj, ready=None, collected=None):
    if ready is None:
        ready = list()
    if collected is None:
        collected = list()
        top = True
    else:
        top = False
    if isinstance(obj, (list, tuple)):
        num_elem = len(obj)
        for idx, inner_obj in enumerate(obj):
            if idx < num_elem - 1:
                new = list(collected)
                new.append(idx)
                get_keys_from_nested(inner_obj, ready=ready, collected=new)
            else:
                collected.append(idx)
                get_keys_from_nested(inner_obj, ready=ready, collected=collected)
    elif isinstance(obj, dict):
        num_elem = len(obj)
        for idx, (k, inner_obj) in enumerate(obj.items()):
            if idx < num_elem - 1:
                new = list(collected)
                new.append(k)
                get_keys_from_nested(inner_obj, ready=ready, collected=new)
            else:
                collected.append(k)
                get_keys_from_nested(inner_obj, ready=ready, collected=collected)
    else:
        ready.append(collected)
    if top:
        return ready


def get_obj_elem_by_path(obj, path):
    for k in path:
        # print('\n')
        # print(k)
        # print(obj)
        obj = obj[k]
    return obj


def write_elem_in_obj_by_path(obj, path, elem):
    for k in path[:-1]:
        obj = obj[k]
    obj[path[-1]] = elem


def flatten(nested):
    output = list()
    if isinstance(nested, (tuple, list)):
        for inner_object in nested:
            flattened = flatten(inner_object)
            output.extend(flattened)
    elif isinstance(nested, (dict, OrderedDict)):
        for inner_object in nested.values():
            flattened = flatten(inner_object)
            output.extend(flattened)
    else:
        output.append(nested)
    return output


def synchronous_flatten(*nested):
    if not isinstance(nested[0], (tuple, list, dict)):
        return [[n] for n in nested]
    output = [list() for _ in nested]
    if isinstance(nested[0], dict):
        for k in nested[0].keys():
            flattened = synchronous_flatten(*[n[k] for n in nested])
            for o, f in zip(output, flattened):
                o.extend(f)
    else:
        try:
            for inner_nested in zip(*nested):
                flattened = synchronous_flatten(*inner_nested)
                for o, f in zip(output, flattened):
                    o.extend(f)
        except TypeError:
            print('(synchronous_flatten)nested:', nested)
            raise
    return output


def loop_through_indices(filename, start_index):
    path, name = split_to_path_and_name(filename)
    if '.' in name:
        inter_list = name.split('.')
        extension = inter_list[-1]
        base = '.'.join(inter_list[:-1])
        base += '#%s'
        name = '.'.join([base, extension])

    else:
        name += '#%s'
    if path != '':
        base_path = os.path.join(path, name)
    else:
        base_path = name
    index = start_index
    while os.path.exists(base_path % index):
        index += 1
    return base_path % index


def add_index_to_filename_if_needed(filename, index=None):
    if index is not None:
        return loop_through_indices(filename, index)
    if os.path.exists(filename):
        return loop_through_indices(filename, 1)
    return filename


def split_to_path_and_name(path):
    parts = path.split('/')
    name = parts[-1]
    path = '/'.join(parts[:-1])
    return path, name


def create_path(path, file_name_is_in_path=False):
    if file_name_is_in_path:
        folder_list = path.split('/')[:-1]
    else:
        folder_list = path.split('/')
    if len(folder_list) > 0:
        if folder_list[0] == '':
            current_folder = '/'
        else:
            current_folder = folder_list[0]
        for idx, folder in enumerate(folder_list):
            if idx > 0:
                current_folder += ('/' + folder)
            if not os.path.exists(current_folder):
                os.mkdir(current_folder)


def add_postfix_to_path_string(path, postfix):
    # print('(add_postfix_to_path_string)path:', path)
    # print('(add_postfix_to_path_string)postfix:', postfix)
    dirs_and_file = path.split('/')
    dirs = dirs_and_file[:-1]
    file_name = dirs_and_file[-1]
    # print('(add_postfix_to_path_string)dirs:', dirs)
    # print('(add_postfix_to_path_string)file_name:', file_name)
    if '.' in file_name:
        name_and_ext = file_name.split('.')
        name = '.'.join(name_and_ext[:-1])
        ext = name_and_ext[-1]
        # print('(add_postfix_to_path_string)ext:', ext)
        new_name = name + postfix + '.' + ext
        # print('(add_postfix_to_path_string)new_name:', new_name)
    else:
        new_name = file_name + postfix
    new_path = '/'.join(dirs + [new_name])
    return new_path


def compute_perplexity(probabilities):
    probabilities[probabilities < 1e-10] = 1e-10
    log_probs = np.log2(probabilities)
    entropy_by_character = np.sum(- probabilities * log_probs, axis=1)
    return np.mean(np.exp2(entropy_by_character))


def compute_loss(predictions, labels):
    predictions[predictions < 1e-10] = 1e-10
    log_predictions = np.log(predictions)
    bpc_by_character = np.sum(- labels * log_predictions, axis=1)
    return np.mean(bpc_by_character)


def compute_bpc(predictions, labels):
    return compute_loss(predictions, labels) / np.log(2)


def compute_accuracy(predictions, labels):
    num_characters = predictions.shape[0]
    num_correct = 0
    for i in range(num_characters):
        if labels[i, np.argmax(predictions, axis=1)[i]] == 1:
            num_correct += 1
    return float(num_correct) / num_characters


def match_two_dicts(small_dict, big_dict):
    """Compares keys of small_dict to keys of big_dict and if in small_dict there is a key missing in big_dict throws
    an error"""
    big_dict_keys = big_dict.keys()
    for key in small_dict.keys():
        if key not in big_dict_keys:
            raise KeyError("Wrong argument name '%s'" % key)
    return True


def split_dictionary(dict_to_split, bases):
    """Function takes dictionary dict_to_split and splits it into several dictionaries according to keys of dicts
    from bases"""
    dicts = list()
    for base in bases:
        if isinstance(base, dict):
            base_keys = base.keys()
        else:
            base_keys = base
        new_dict = dict()
        for key, value in dict_to_split.items():
            if key in base_keys:
                new_dict[key] = value
        dicts.append(new_dict)
    return dicts


def link_into_dictionary(old_dictionary, old_keys, new_key):
    """Used in _parse_train_method_arguments to united several kwargs into one dictionary
    Args:
        old_dictionary: a dictionary which entries are to be united
        old_keys: list of keys to be united
        new_key: the key of new entry  containing linked dictionary"""
    linked = dict()
    for old_key in old_keys:
        if old_key in linked:
            linked[old_key] = old_dictionary[old_key]
            del old_dictionary[old_key]
    old_dictionary[new_key] = linked
    return old_dictionary


def paste_into_nested_structure(structure, searched_key, value_to_paste):
    # print('***********************')
    if isinstance(structure, dict):
        for key, value, in structure.items():
            # print('key:', key)
            if key == searched_key:
                structure[key] = construct(value_to_paste)
            else:
                if isinstance(value, (dict, list, tuple)):
                    paste_into_nested_structure(value, searched_key, value_to_paste)
    elif isinstance(structure, (list, tuple)):
        for elem in structure:
            paste_into_nested_structure(elem, searched_key, value_to_paste)


def check_if_key_in_nested_dict(dictionary, keys):
    new_key_list = keys[1:]
    if keys[0] not in dictionary:
        return False
    if len(new_key_list) == 0:
        return True
    value = dictionary[keys[0]]
    if not isinstance(value, dict):
        return False
    return check_if_key_in_nested_dict(value, new_key_list)


def search_in_nested_dictionary(dictionary, searched_key):
    for key, value in dictionary.items():
        if key == searched_key:
            return value
        else:
            if isinstance(value, dict):
                returned_value = search_in_nested_dictionary(value, searched_key)
                if returned_value is not None:
                    return returned_value
    return None


def add_missing_to_list(extended_list, added_list):
    for elem in added_list:
        if elem not in extended_list:
            extended_list.append(elem)
    return extended_list


def print_and_log(*inputs, log=True, _print=True, fn=None):
    if _print:
        print(*inputs)
    if log:
        for inp in inputs:
            with open(fn, 'a') as fd:
                fd.write(str(inp))
        with open(fn, 'a') as fd:
            fd.write('\n')


def apply_temperature(array, axis, temperature):
    array = np.power(array, 1/temperature)
    norm = np.sum(array, axis=axis, keepdims=True)
    return array / norm


def compute_num_of_repeats(start_axis, last_axis_plus_1, removed_axis, shape):
    num_of_repeats = 1
    for ax_num in range(start_axis, last_axis_plus_1):
        if ax_num != removed_axis:
            num_of_repeats *= shape[ax_num]
    return num_of_repeats


def construct_indices(constructed_axis, removed_axis, shape):
    num_one_value_repeats = compute_num_of_repeats(constructed_axis+1, len(shape), removed_axis, shape)
    num_pattern_repeats = compute_num_of_repeats(0, constructed_axis, removed_axis, shape)
    pattern = list()
    for i in range(shape[constructed_axis]):
        pattern.extend([i] * num_one_value_repeats)
    pattern = pattern * num_pattern_repeats
    return np.array(pattern)


def sample(array, axis):
    shape = array.shape
    if axis == -1:
        axis = len(shape) - 1
    c = array.cumsum(axis=axis)
    r_shape = list(shape)
    r_shape[axis] = 1
    r = np.random.rand(*r_shape)
    mask = (r < c).argmax(axis)
    axis_indices = np.reshape(mask, (-1))
    all_indices = list()
    for ax_num in range(len(shape)):
        if ax_num != axis:
            all_indices.append(construct_indices(ax_num, axis, shape))
        else:
            all_indices.append(axis_indices)
    answer = np.zeros(shape)
    for ax_num in range(len(shape)):
        exec("i%s = all_indices[%s]" % (ax_num, ax_num))
    init_string = ('answer[' + 'i%s' + ', i%s'*(len(shape)-1) + '] = 1.') % tuple([i for i in range(len(shape))])
    exec(init_string)
    return answer


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpu_names = sorted(gpu_names, key=lambda elem: int(elem.split(':')[-1]))
    return gpu_names


def device_name_scope(device_name):
    return device_name[1:4] + device_name[5:]


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    with tf.name_scope('average_gradients'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


def average_gradients_not_balanced(tower_grads, num_active):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    with tf.name_scope('not_balanced_average'):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_sum(grad, 0) / sum(num_active)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


def get_num_gpus_and_bs_on_gpus(batch_size, num_gpus, num_available_gpus):
    batch_sizes_on_gpus = list()
    if num_available_gpus < num_gpus:
        print("Can't build model on %s gpus: only %s gpus are available.\n\tFalling to %s gpus" %
              (num_gpus, num_available_gpus, num_available_gpus))
        num_gpus = num_available_gpus
    if num_gpus > batch_size:
        num_gpus = batch_size
        print("Can't build model on %s gpus: batch size = %s (batch size is to small).\n\tFalling to %s gpus" %
              (num_gpus, batch_size, batch_size))
    else:
        num_gpus = num_gpus
    if num_gpus > 1:
        gpu_batch = batch_size // num_gpus
        num_remaining_elements = batch_size
        for _ in range(num_gpus - 1):
            batch_sizes_on_gpus.append(gpu_batch)
            num_remaining_elements -= gpu_batch
        batch_sizes_on_gpus.append(num_remaining_elements)
    else:
        batch_sizes_on_gpus = [batch_size]
    return num_gpus, batch_sizes_on_gpus


def nested2string(nested, indent=0):
    string = list()
    indent = indent
    recur_nested_2_string(nested, string, indent)
    string = ''.join(string)
    return string


def recur_nested_2_string(nested, string, indent):
    ind = ' ' * indent
    if not isinstance(nested, (list, tuple, dict)):
        if isinstance(nested, str):
            if len(nested) > 50:
                string.append('VERY_LONG_STRING')
            else:
                string.append('\"' + add_escape_characters(nested) + '\"')
        else:
            string.append(str(nested))
    elif isinstance(nested, dict):
        string.append('{')
        if len(nested) > 0:
            string.append('\n')
        for idx, (key, value) in enumerate(nested.items()):
            if isinstance(key, str):
                string.append(ind + ' ' *4 + '\"' + key + '\"' + ': ')
            else:
                string.append(ind + ' ' *4 + str(key) + ': ')
            recur_nested_2_string(value, string, indent + 4)
            if idx < len(nested) - 1:
                string.append(',\n')
        string.append('}')
    elif isinstance(nested, list):
        string.append('[')
        if len(nested) > 0:
            string.append('\n')
        for idx, value in enumerate(nested):
            string.append(ind + ' ' *4)
            recur_nested_2_string(value, string, indent + 4)
            if idx < len(nested) - 1:
                string.append(',\n')
        string.append(']')

    elif isinstance(nested, tuple):
        string.append('(')
        if len(nested) > 0:
            string.append('\n')
        else:
            string.append(',')
        for idx, value in enumerate(nested):
            string.append(ind + ' ' *4)
            recur_nested_2_string(value, string, indent + 4)
            if idx < len(nested) - 1:
                string.append(',\n')
        string.append(')')


def add_escape_characters(string):
    new_string = ''
    for char in string:
        if char in escape_sequences and char not in skipped_escape_sequences:
            new_string += escape_sequences_replacements[char]
        else:
            new_string += char
    return new_string


def all_entries_in_nested_structure(nested):
    all_entries = list()
    recur_entries(nested, all_entries)
    return all_entries


def recur_entries(nested, all_entries):
    if isinstance(nested, dict):
        all_entries.extend(list(nested.keys()))
        for value in nested.values():
            recur_entries(value, all_entries)
    elif isinstance(nested, (list, tuple)):
        for value in nested:
            recur_entries(value, all_entries)


def unite_dicts(list_of_dicts):
    new_dict = OrderedDict()
    for d in list_of_dicts:
        new_dict.update(d)
    return new_dict


def unite_nested_dicts(list_of_nested, depth):
    if isinstance(list_of_nested[0], dict):
        res = dict()
    elif isinstance(list_of_nested[0], OrderedDict):
        res = OrderedDict()
    else:
        raise InvalidArgumentError(
            "list_of_nested has to be list of dicts o list of OrderedDicts",
            list_of_nested,
            'list_of_nested',
            "list of dict or OrderedDicts"
        )
    if depth < 1:
        for d in list_of_nested:
            res.update(d)
    else:
        for k in list_of_nested[0].keys():
            res[k] = unite_nested_dicts(
                [l[k] for l in list_of_nested],
                depth-1
            )
    return res


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#
# def write_equation(a_ndims, b_ndims):
#     letters = 'ijklmnoprs'
#     out_ndims = a_ndims + b_ndims - 3
#     var_set_indices = letters[0]
#     a_indices = letters[1:a_ndims - 2]
#     b_indices = letters[out_ndims - b_ndims + 1:out_ndims - 2]
#     last_indices = letters[out_ndims - 2: out_ndims + 1]
#     a_str = var_set_indices + a_indices + last_indices[:2]
#     b_str = var_set_indices + b_indices + last_indices[1:]
#     out_str = var_set_indices + a_indices + b_indices + last_indices[0] + last_indices[2]
#     return a_str + ',' + b_str + '->' + out_str


def write_equation(a_ndims, b_ndims, base_ndims):
    letters = 'ijklmnoprs'
    # out_ndims = a_ndims + base_ndims[1] - 2
    a_set_ndims = a_ndims - base_ndims[0]
    b_set_ndims = b_ndims - base_ndims[1]
    a_short = a_set_ndims == 0
    b_short = b_set_ndims == 0

    var_set_ndims = max(a_set_ndims, b_set_ndims)
    if var_set_ndims == 0:
        var_set_ndims = b_ndims - base_ndims[1]
    var_set_indices = letters[0:var_set_ndims]
    a_extended_ndims = a_ndims - a_set_ndims - 2
    b_extended_ndims = b_ndims - b_set_ndims - 2
    a_indices = letters[var_set_ndims:var_set_ndims + a_extended_ndims]
    b_indices = letters[var_set_ndims + a_extended_ndims:var_set_ndims + a_extended_ndims + b_extended_ndims]
    not_last_ndims = var_set_ndims + a_extended_ndims + b_extended_ndims
    last_indices = letters[not_last_ndims: not_last_ndims + 3]
    if not a_short:
        a_str = str(var_set_indices)
    else:
        a_str = ''
    if not b_short:
        b_str = str(var_set_indices)
    else:
        b_str = ''
    a_str += a_indices + last_indices[:2]
    b_str += b_indices + last_indices[1:]
    out_str = var_set_indices + a_indices + b_indices + last_indices[0] + last_indices[2]
    return a_str + ',' + b_str + '->' + out_str


def custom_matmul(a, b, base_ndims=None, eq=None, name_scope='custom_matmul'):
    """Special matmul for several exercises simultaneous processing.
    Matrix multiplication is performed across 2 last dimensions of a and b. Either mapping or broadcasting is performed
    across all dimensions except for last 2. Specifically if (a_ndims > base_ndims[0] and b_ndims > base_ndims[1])
    and (a_ndims - base_ndims[0] == b_ndims - base_ndims[1]) first a_ndims - base_ndims[0] dimensions of a and b are
    mapped:
        consider base_ndims = [3,2], a_ndim = 5, b_ndim = 4 than eq = 'ijklm,ijmn->ijkln'.
    If base_ndims[i] > 2 than mutual broadcasting is performed. In case when base_ndims[0] > 2 base_ndims[1] > 2
    this means outer product. a dimensions are put in front of b dimensions. Examples:
        base_ndims = [3,3]    eq = 'ijkmn,ijlno->ijklmo'
        base_ndims = [3,4]    eq = 'ijkno,ijlmop->ijklmnp'"""
    with tf.name_scope(name_scope):
        if eq is not None:
            res = tf.einsum(eq, a, b)
        else:
            if base_ndims is None:
                base_ndims = [2, 2]
            a_shape = a.get_shape().as_list()
            b_shape = b.get_shape().as_list()
            a_ndims = len(a_shape)
            b_ndims = len(b_shape)
            if a_ndims < base_ndims[0] or b_ndims < base_ndims[1] or \
                    (a_ndims > base_ndims[0] and b_ndims > base_ndims[1] and
                                 a_ndims - base_ndims[0] != b_ndims - base_ndims[1]):
                raise InvalidArgumentError(
                    'if len(a.shape) - base_ndims[0] > 0 and len(b.shape) - base_ndims[1] > 0 than '
                    'a and b have to satisfy condition \n'
                    'len(a.shape) - base_ndims[0] == len(b.shape) - base_ndims[1]\n'
                    'whereas\n'
                    'a.shape = %s\n'
                    'b.shape = %s\n'
                    'base_ndims = %s' % (a_shape, b_shape, base_ndims),
                    [a, b],
                    'tensors a and b',
                    'tensors which satisfy \n'
                    'len(a.shape) - base_ndims[0] == len(b.shape) - base_ndims[1]')
            else:
                eq = write_equation(a_ndims, b_ndims, base_ndims)
                res = tf.einsum(eq, a, b)
    return res


def custom_add(a, b, base_ndims=None, name='custom_add'):
    """Special addition for several exercises simultaneous processing.
    Across last base_ndims[0] of a and last base_ndims[1] of b usual addition (with broadcasting if needed) is
     performed.
    If a_ndims > base_ndims[0] and b_ndims > base_ndims[1] and a_ndims - base_ndims[0] == b_ndims - base_ndims[1]
    first a_ndims - base_ndims[0] are mapped meaning that addition is performed between corresponding dims. If one of
    tensors args[i] satisfies condition len(args[i].shape) == base_ndims[i] whereas the other does not satisfies it
    broadcasting to len(args[i].shape) - base_ndims[i] of args[i] is performed."""
    with tf.name_scope('custom_add'):
        if base_ndims is None:
            base_ndims = [2, 1]
        a_shape = a.get_shape().as_list()
        b_shape = b.get_shape().as_list()
        a_ndims = len(a_shape)
        b_ndims = len(b_shape)
        b_is_broadcasted = base_ndims[0] > base_ndims[1]
        max_ndims = max(a_ndims, b_ndims)
        biggest_base = max(base_ndims[0], base_ndims[1])
        smallest_base = min(base_ndims[0], base_ndims[1])
        diff = biggest_base - smallest_base
        if a_ndims > base_ndims[0] and b_ndims > base_ndims[1]:
            if a_ndims - base_ndims[0] == b_ndims - base_ndims[1]:
                mapped_ndims = a_ndims - base_ndims[0]
                longest_indices = [i for i in range(max_ndims)]
                forward_perm = longest_indices[mapped_ndims:max_ndims-smallest_base] + \
                               longest_indices[:mapped_ndims] + longest_indices[max_ndims-smallest_base:]
                backward_perm = longest_indices[diff:diff+mapped_ndims] + longest_indices[:diff] \
                                + longest_indices[max_ndims-smallest_base:]
                # print(forward_perm)
                # print(backward_perm)
                if b_is_broadcasted:
                    a_tr = tf.transpose(a, perm=forward_perm)
                    res_tr = a_tr + b
                    res = tf.transpose(res_tr, perm=backward_perm, name=name)
                elif base_ndims[0] == base_ndims[1]:
                    res = tf.add(a, b, name=name)
                else:
                    b_tr = tf.transpose(b, perm=forward_perm)
                    res_tr = b_tr + a
                    res = tf.transpose(res_tr, perm=backward_perm, name=name)
            else:
                raise InvalidArgumentError(
                    'if len(a.shape) - base_ndims[0] > 0 and len(b.shape) - base_ndims[1] > 0 than '
                    'a and b have to satisfy condition \n'
                    'len(a.shape) - base_ndims[0] == len(b.shape) - base_ndims[1]',
                    [a, b],
                    'tensors a and b',
                    'tensors which satisfy \n'
                    'len(a.shape) - base_ndims[0] == len(b.shape) - base_ndims[1]')
        else:
            res = tf.add(a, b, name=name)
        return res


def load_vocabulary(vocabulary_path):
    with open(vocabulary_path, 'r') as f:
        lines = f.read().split('\n')
    return [ast.literal_eval(l) for l in lines]


def stop_gradient_in_nested(nested):
    paths = get_keys_from_nested(nested)
    for path in paths:
        write_elem_in_obj_by_path(nested, path, tf.stop_gradient(get_obj_elem_by_path(nested, path)))
    return nested


def extract_op_name(full_name):
    scopes_stripped = full_name.split('/')[-1]
    return scopes_stripped.split(':')[0]


def compose_save_list(*pairs, name_scope='save_list'):
    with tf.name_scope(name_scope):
        save_list = list()
        for pair in pairs:
            # print('pair:', pair)
            [variables, new_values] = synchronous_flatten(pair[0], pair[1])
            # print("(useful_functions.compose_save_list)variables:", variables)
            # variables = flatten(pair[0])
            # # print(variables)
            # new_values = flatten(pair[1])
            for variable, value in zip(variables, new_values):
                name = extract_op_name(variable.name)
                save_list.append(tf.assign(variable, value, name='assign_save_%s' % name))
        return save_list


def compose_reset_list(*args, name_scope='reset_list'):
    with tf.name_scope(name_scope):
        reset_list = list()
        flattened = flatten(args)
        for variable in flattened:
            shape = variable.get_shape().as_list()
            name = extract_op_name(variable.name)
            reset_list.append(tf.assign(variable, tf.zeros(shape), name='assign_reset_%s' % name))
        return reset_list


def compose_randomize_list(*args, name_scope='randomize_list'):
    with tf.name_scope(name_scope):
        randomize_list = list()
        flattened = flatten(args)
        for variable in flattened:
            shape = variable.get_shape().as_list()
            name = extract_op_name(variable.name)
            assign_tensor = tf.truncated_normal(shape, stddev=1.)
            # assign_tensor = tf.Print(assign_tensor, [assign_tensor], message='assign tensor:')
            assign = tf.assign(variable, assign_tensor, name='assign_reset_%s' % name)
            randomize_list.append(assign)
        return randomize_list


def block_diagonal(matrices, dtype=tf.float32):
    r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

    Args:
        matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
        matrices with the same batch dimension).
        dtype: Data type to use. The Tensors in `matrices` must match this dtype.
    Returns:
        A matrix with the input matrices stacked along its main diagonal, having
        shape [..., \sum_i N_i, \sum_i M_i].
    """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                 [(row_before_length, row_after_length)]],
                axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked


def retrieve_from_inner_dicts(d, key):
    map_ = dict()
    retrieved = list()
    start = 0
    for k, v in d.items():
        if key in v:
            if isinstance(v[key], list):
                map_[k] = [start, start + len(v[key])]
                start += len(v[key])
                retrieved.extend(v[key])
            else:
                map_[k] = start
                start += 1
                retrieved.append(v[key])
        else:
            print("WARNING: %s misses %s (retrieve_from_inner_dicts)" % (k, key))
    return retrieved, map_


def distribute_into_inner_dicts(d, key, to_distribute, map_):
    for k, v in d.items():
        if k in map_:
            if isinstance(map_[k], list):
                v[key] = to_distribute[map_[k][0]:map_[k][1]]
            else:
                v[key] = to_distribute[map_[k]]
        else:
            print("WARNING: %s misses %s (distribute_into_inner_dicts)" % (k, key))
    return d


def print_optimizer_ins(opt_ins):
    """Help method for debugging meta optimizers"""
    for idx, (ok, ov) in enumerate(opt_ins.items()):
        print(ok)
        for ik, iv in ov.items():
            print('')
            print(ik)
            print(iv)
        if idx < len(opt_ins) - 1:
            print('\n' * 2)
    print('*' * 20)
    print('\n' * 2)


def create_distribute_map(num_distributed, result_length):
    div = result_length // num_distributed
    mod = result_length % num_distributed
    if num_distributed < result_length:
        num_repeats = [div] * num_distributed
        for i in range(mod):
            num_repeats[i] += 1
    else:
        num_repeats = [1] * result_length
    map_ = list(itertools.chain(*[[i] * n_rep for i, n_rep in enumerate(num_repeats)]))
    return map_


def remove_keys_from_dictionary(d, keys):
    for key in keys:
        if key in d:
            del d[key]
    return d


def extend_dictionary(dictionary, key_path):
    d = dictionary
    if key_path is None:
        key_path = []
    for key in key_path:
        if key not in d:
            d[key] = dict()
        d = d[key]
    return d


def construct_dict_without_none_entries(dictionary):
    res = dict()
    for k, v in dictionary.items():
        if v is not None:
            res[k] = v
    return res


def values_from_nested(nested):
    accumulated_values = list()
    if isinstance(nested, dict):
        for v in nested.values():
            if isinstance(v, (dict, list, tuple)):
                accumulated_values.extend(values_from_nested(v))
            else:
                accumulated_values.append(v)
    elif isinstance(nested, (list, tuple)):
        for v in nested:
            if isinstance(v, (dict, list, tuple)):
                accumulated_values.extend(values_from_nested(v))
            else:
                accumulated_values.append(v)
    return accumulated_values


def l2_loss_per_elem(t):
    shape = t.get_shape().as_list()
    num_elems = 1
    for dim in shape:
        num_elems *= dim
    return tf.nn.l2_loss(t) / num_elems


def apply_recur(nested, func, depth, obj_types):
    if isinstance(nested, list):
        res = list()
        for elem in nested:
            res.append(apply_to_nested_on_depth(elem, func, depth=depth, obj_types=obj_types))
    elif isinstance(nested, tuple):
        res = list()
        for elem in nested:
            res.append(apply_to_nested_on_depth(elem, func, depth=depth, obj_types=obj_types))
        res = tuple(res)
    elif isinstance(nested, dict):
        res = dict()
        for k, v in nested.items():
            res[k] = apply_to_nested_on_depth(v, func, depth=depth, obj_types=obj_types)
    elif isinstance(nested, OrderedDict):
        res = OrderedDict()
        for k, v in nested.items():
            res[k] = apply_to_nested_on_depth(v, func, depth=depth, obj_types=obj_types)
    else:
        raise InvalidArgumentError("nested has to be object of types list, tuple, dict, OrderedDict",
                                   nested,
                                   'nested',
                                   "list, tuple, dict, OrderedDict")
    return res


def apply_to_nested_on_depth(nested, func, depth=None, obj_types=None):
    if obj_types is None:
        obj_types = (list, tuple, dict, OrderedDict)
    if depth is None:
        if isinstance(nested, obj_types):
            res = apply_recur(nested, func, depth, obj_types)
        else:
            res = func(nested)
    else:
        if depth > 0:
            res = apply_recur(nested, func, depth-1, obj_types)
        else:
            res = func(nested)
    return res


def tf_print_nested(nested, nested_name, input_, summarize, path=None):
    if path is None:
        path = []
    with tf.device('/cpu:0'):
        if isinstance(nested, (list, tuple)):
            for idx, elem in enumerate(nested):
                input_ = tf_print_nested(elem, nested_name, input_, summarize, path + [idx])
        elif isinstance(nested, dict):
            for k, v in nested.items():
                input_ = tf_print_nested(v, nested_name, input_, summarize, path + [k])
        else:
            input_ = tf.Print(
                input_,
                [nested],
                message=nested_name + ''.join(['[%s]']*len(path)) % tuple(path) + ' = ',
                summarize=summarize
            )
    return input_


def cum_and(l, name_scope='list_summation'):
    with tf.name_scope(name_scope):
        res = True
        for elem in l:
            res = res and elem
        return res


def sort_lists_map(lists):
    list_len = len(lists[0])
    values = [list() for _ in lists[0]]
    for l in lists:
        for idx, v in enumerate(l):
            if v not in values[idx]:
                values[idx].append(v)
    new_values = list()
    for v in values:
        both_str_and_ints = not (cum_and([isinstance(elem, str) for elem in v]) or
                                 cum_and([isinstance(elem, int) for elem in v]))
        if both_str_and_ints:
            new_values.append(sorted(v, key=lambda elem: str(elem) if isinstance(elem, int) else elem))
        else:
            new_values.append(sorted(v))
    values = new_values
    periods = [1] * list_len
    for idx in range(list_len-2, -1, -1):
        periods[idx] = periods[idx+1] * len(values[idx+1])
    return values, periods


def sort_lists_of_ints_and_str(lists):
    values, periods = sort_lists_map(lists)

    def key_func(elem):
        res = 0
        for idx, v in enumerate(elem):
            res += values[idx].index(v) * periods[idx]
        return res
    return sorted(lists, key=key_func)


def go_through_nested_with_name_scopes_to_perform_func(nested, remaining_key_values, path, func):
    if len(remaining_key_values) > 0:
        for key in remaining_key_values[0]:
            p = path + [key]
            with tf.name_scope(str(key)):
                go_through_nested_with_name_scopes_to_perform_func(nested, remaining_key_values[1:], p, func)
    else:
        write_elem_in_obj_by_path(nested, path, func(get_obj_elem_by_path(nested, path)))


def go_through_nested_with_name_scopes_to_perform_func_and_distribute_results(
        nested, remaining_key_values, path, func, results):
    # print("(go_through_nested_with_name_scopes_to_perform_func_and_distribute_results)remaining_key_values:",
    #       remaining_key_values)
    # print("(go_through_nested_with_name_scopes_to_perform_func_and_distribute_results)path:",
    #       path)
    # print("(go_through_nested_with_name_scopes_to_perform_func_and_distribute_results)results:",
    #       results)
    if len(remaining_key_values) > 0:
        for key in remaining_key_values[0]:
            p = path + [key]
            # print("(go_through_nested_with_name_scopes_to_perform_func_and_distribute_results)key:", key)
            with tf.name_scope('ns_' + str(key)):
                go_through_nested_with_name_scopes_to_perform_func_and_distribute_results(
                    nested, remaining_key_values[1:], p, func, results)
    else:
        for res, t in zip(results, func(get_obj_elem_by_path(nested, path))):
            # print("(go_through_nested_with_name_scopes_to_perform_func_and_distribute_results)t:", t)
            write_elem_in_obj_by_path(res, path, t)


def global_l2_loss(tensor_list, name_scope='global_l2_loss'):
    with tf.name_scope(name_scope):
        loss = 0
        for t in tensor_list:
            loss += tf.nn.l2_loss(t)
        return loss


def global_norm(tensor_list, name_scope='global_norm'):
    with tf.name_scope(name_scope):
        return tf.sqrt(global_l2_loss(tensor_list))


def filter_none_gradients(grads_and_vars):
    res = list()
    for gv in grads_and_vars:
        if gv[0] is not None:
            res.append(gv)
    return res


def func_on_list_in_nested(nested, func):
    """ Goes up to first list and stacks it. nested is a nested dictionary"""
    for k, v in nested.items():
        if isinstance(v, list):
            nested[k] = func(v)
        else:
            nested[k] = func_on_list_in_nested(v, func)
    return nested


def append_to_nested(result, to_append):
    """result and to_append are nested dicts"""
    for k, v in to_append.items():
        if isinstance(v, dict):
            append_to_nested(result[k], v)
        else:
            result[k].append(v)
    return result


def get_average_with_weights_func(weights):
    sum_ = float(sum(weights))
    prep_weights = [w / sum_ for w in weights]
    def average_func(values):
        res = 0
        for w, v in zip(prep_weights, values):
            res += w * v
        return res
    return average_func


def nth_element_of_sequence_of_sequences(s, n):
    res = list()
    for el in s:
        res.append(el[n])
    return res


def convert(value, type_):
    try:
        # Check if it's a builtin type
        module = importlib.import_module('builtins')
        cls = getattr(module, type_)
    except AttributeError:
        # if not, separate module and class
        module, type_ = type_.rsplit(".", 1)
        module = importlib.import_module(module)
        cls = getattr(module, type_)
    return cls(value)


def get_hps(file_name):
    res = dict()
    with open(file_name, 'r') as f:
        lines = f.read().split('\n')
        hp_names = lines[0].split()
        hp_types = lines[1].split()
        for hp_name, hp_type, line in zip(hp_names, hp_types, lines[2:]):
            res[hp_name] = [convert(v, hp_type) for v in line.split()]
    return res


def get_combs_and_num_exps(eval_dir, hp_order):
    hp_sets = list()
    if os.path.exists(eval_dir):
        contents = os.listdir(eval_dir)
        exp_description_files = list()
        for entry in contents:
            if entry[-4:] == '.txt' and is_int(entry[:-4]):
                exp_description_files.append(entry)
        exp_description_files = sorted(exp_description_files, key=lambda elem: int(elem[:-4]))

        hp_layout_file = os.path.join(eval_dir, 'hp_layout.txt')
        with open(hp_layout_file, 'r') as f:
            layout = remove_empty_strings_from_list(remove_empty_strings_from_list(f.read().split('\n'))[0].split())
        # print("(useful_functions.get_combs_and_num_exps)len(exp_description_files):", len(exp_description_files))
        # print("(useful_functions.get_combs_and_num_exps)exp_description_files:", exp_description_files)
        map_ = dict(list(zip(hp_order, [layout.index(hp) for hp in hp_order])))
        file_is_broken = False
        for file_name in exp_description_files:
            file_path = os.path.join(eval_dir, file_name)
            with open(file_path, 'r') as f:
                lines = f.read().split('\n')
                if len(lines) >= 2:
                    types = lines[1].split()
                    hp_set = list()
                    values = [convert(v, t) for v, t in zip(lines[0].split(), types)]
                    for hp in hp_order:
                        hp_set.append(values[map_[hp]])
                    hp_set = tuple(hp_set)
                    if hp_set not in hp_sets:
                        hp_sets.append(hp_set)
                else:
                    file_is_broken = True
            if file_is_broken:
                file_is_broken = False
                print("WARNING: file %s is broken" % file_path)
        # print("(useful_functions.get_combs_and_num_exps)hp_sets:", hp_sets)

        if len(exp_description_files) > 0:
            last_file_name = exp_description_files[-1]
        else:
            last_file_name = None
        return hp_sets, len(exp_description_files), last_file_name
    else:
        return [], 0, None


def parse_header_line(line):
    names = line.split()
    metrics = list()
    hp_names = list()
    for name in names:
        if name in METRICS:
            metrics.append(name)
        else:
            hp_names.append(name)
    return metrics, hp_names


def check_if_line_contains_results(line):
    if len(line) == 0:
        return False
    for c in line:
        if c not in NUMBER_CHARS:
            return False
    return True


def check_if_line_is_header(line):
    return not check_if_line_contains_results(line)


def get_type_simple(value):
    if '.' in value or 'e' in value:
        value_type = 'float'
    else:
        value_type = 'int'
    return value_type


def extract_hp_set(line, hp_names):
    hp_set = dict()
    hp_values = line.split()[-len(hp_names):]
    # print("(useful_functions.extract_hp_set)hp_values:", hp_values)
    # print("(useful_functions.extract_hp_set)hp_names:", hp_names)
    for hp_name, hp_value in zip(hp_names, hp_values):
        hp_set[hp_name] = convert(hp_value, get_type_simple(hp_value))
    return hp_set


def extract_metrics(line, metric_names):
    res_set = dict()
    hp_values = line.split()[:len(metric_names)]
    # print("(useful_functions.extract_hp_set)hp_values:", hp_values)
    # print("(useful_functions.extract_hp_set)hp_names:", hp_names)
    for name, value in zip(metric_names, hp_values):
        res_set[name] = convert(value, get_type_simple(value))
    return res_set


def get_combs_and_num_exps_pupil(eval_file, order):
    tested_combs = list()
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            lines = f.read().split('\n')
        # print("(useful_functions.get_combs_and_num_exps_pupil)lines:", lines)
        metrics, hp_names = parse_header_line(lines[0])
        for line in lines[1:]:
            if check_if_line_contains_results(line):
                hp_set = extract_hp_set(line, hp_names)
                # print("(useful_functions.get_combs_and_num_exps_pupil)hp_set:", hp_set)
                comb = list()
                for name in order:
                    comb.append(hp_set[name])
                tested_combs.append(comb)
        return tested_combs, len(tested_combs)
    else:
        return [], 0


def get_exp_description_files_and_contents(eval_dir):
    contents = os.listdir(eval_dir)
    exp_description_files = list()
    for entry in contents:
        if entry[-4:] == '.txt' and is_int(entry[:-4]):
            exp_description_files.append(entry)
    return exp_description_files, contents


def get_num_exps_and_res_files(eval_dir):
    pairs = list()
    if os.path.exists(eval_dir):
        exp_description_files, contents = get_exp_description_files_and_contents(eval_dir)
        exp_description_files = sorted(exp_description_files, key=lambda elem: int(elem[:-4]))
        biggest_idx = 0
        for file_name in exp_description_files:
            if int(file_name[:-4]) > biggest_idx:
                biggest_idx = int(file_name[:-4])
            if file_name[:-4] in contents:
                pairs.append(
                    (os.path.join(eval_dir, file_name), os.path.join(eval_dir, file_name[:-4]))
                )
            else:
                print("WARNING: missing results directory for experiment %s" % file_name)
        return len(pairs), biggest_idx, pairs
    else:
        return 0, None, []


def remove_repeats_from_list(l):
    res = list()
    for v in l:
        if v not in res:
            res.append(v)
    return res


def compose_hp_confs(file_name, eval_dir_or_file, chop_last_experiment=False, model='optimizer'):
    grids, init_conf, num_exps = make_initial_grid(
        file_name,
        eval_dir_or_file,
        chop_last_experiment=chop_last_experiment,
        model=model
    )
    confs = list()
    # print("(useful_functions.compose_hp_confs)model:", model)
    for grid in grids:
        # print("(useful_functions.compose_hp_confs)grid:", grid)
        confs.extend(form_confs(grid, init_conf))
    return confs, num_exps


def get_hp_names_from_conf_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.read().split('\n')
    hp_names = remove_empty_strings_from_list(lines[0].split())
    return hp_names


def read_text_conf_file(file_name):
    init_conf = OrderedDict()
    init_grid_values = list()
    with open(file_name, 'r') as f:
        lines = f.read().split('\n')
    hp_names = lines[0].split()
    hp_types = lines[1].split()
    num_params = len(hp_names)
    for hp_name, hp_type, line in zip(hp_names, hp_types, lines[2:2 + num_params]):
        param_values = remove_repeats_from_list([convert(v, hp_type) for v in line.split()])
        init_conf[hp_name] = param_values
        init_grid_values.append(param_values)
    if len(lines) > 2 + num_params and len(lines[2 + num_params]) > 0:
        num_repeats = int(lines[2 + num_params])
    else:
        num_repeats = 1
    return hp_names, init_conf, init_grid_values, num_repeats


def сreate_grids_after_file_parsing(
        hp_names,
        init_conf,
        init_grid_values,
        num_repeats,
        eval_dir_or_file,
        chop_last_experiment=False,
        model='optimizer'
):
    if model == 'optimizer':
        tested_combs, num_exps, last_exp_file_name = get_combs_and_num_exps(eval_dir_or_file, hp_names)
        # print("(useful_functions.make_initial_grid)tested_combs:", tested_combs)
    else:
        tested_combs, num_exps = get_combs_and_num_exps_pupil(eval_dir_or_file, hp_names)
        last_exp_file_name = None
        # print("(useful_functions.make_initial_grid)tested_combs:", tested_combs)
    if num_exps > 0 and chop_last_experiment and model == 'optimizer':
        os.remove(os.path.join(eval_dir_or_file, last_exp_file_name))
        shutil.rmtree(os.path.join(eval_dir_or_file, last_exp_file_name[:-4]))
        tested_combs = tested_combs[:-1]
        num_exps -= 1
    # print("(useful_functions.make_initial_grid)tested_combs:", tested_combs)
    exp_counter_grid = np.zeros(tuple([len(v) for v in init_conf.values()]))
    # print("(useful_functions.make_initial_grid)init_grid_values:", init_grid_values)
    # print("(useful_functions.make_initial_grid)hp_names:", hp_names)
    for tested_comb in tested_combs:
        indices = list()
        for p_idx, v in enumerate(tested_comb):
            try:
                indices.append(init_grid_values[p_idx].index(v))
            except ValueError:
                print("(useful_functions.make_initial_grid)init_grid_values:", init_grid_values)
                print("(useful_functions.make_initial_grid)p_idx:", p_idx)
                print("(useful_functions.make_initial_grid)tested_comb:", tested_comb)
                print("WARNING: UNKNOWN POINT")
        exp_counter_grid[tuple(indices)] += 1.
    grids = slice_to_conf_grids(exp_counter_grid, num_repeats), init_conf, num_exps
    return grids


def make_initial_grid(file_name, eval_dir_or_file, chop_last_experiment=False, model='optimizer'):
    hp_names, init_conf, init_grid_values, num_repeats = read_text_conf_file(file_name)
    grids = сreate_grids_after_file_parsing(
        hp_names,
        init_conf,
        init_grid_values,
        num_repeats,
        eval_dir_or_file,
        chop_last_experiment=chop_last_experiment,
        model=model,
    )
    return grids


def slice_to_conf_grids(exp_counter_grid, num_repeats):
    grids = list()
    for i in range(num_repeats):
        mask = np.greater(exp_counter_grid, i)
        grids.append(mask.astype(float))
    return grids


def one_dim_idx_2_multidim_indices(idx, shape):
    indices = list()
    quotient = idx
    for dim in shape[::-1]:
        indices.append(quotient % dim)
        quotient = quotient // dim
    indices.reverse()
    return indices


def get_missing_entries(grid):
    # print("(useful_functions.get_missing_entries)grid:", grid)
    missing = list()
    sh = grid.shape
    # print("(useful_functions.get_missing_entries)sh:", sh)
    # print("(useful_functions.get_missing_entries)sh:", sh)
    if len(sh) == 0:
        return []
    num_entries = reduce(lambda x, y: x*y, sh)
    for entry_num in range(num_entries):
        indices = one_dim_idx_2_multidim_indices(entry_num, sh)
        # if entry_num in [191, 192, 193, 194]:
        #     print("(useful_functions.get_missing_entries)indices on entry_num %s:" % entry_num, indices) 
        if grid[tuple(indices)] == 0:
            missing.append(indices)
        # else:
        # else:
        #     print("(useful_functions.get_missing_entries)present indices:", indices)
    # print("(useful_functions.get_missing_entries)grid[(0, 12, 11)]:", grid[(0, 12, 11)])
    # print("(useful_functions.get_missing_entries)grid[(0, 12, 12)]:", grid[(0, 12, 12)])
    # print("(useful_functions.get_missing_entries)grid[(0, 12, 13)]:", grid[(0, 12, 13)])
    # print("(useful_functions.get_missing_entries)grid[(0, 12, 14)]:", grid[(0, 12, 14)])
    return missing


def get_missing_hp_sets(conf_file, eval_dir, model):
    missing_hp_sets = list()
    grids, conf, _ = make_initial_grid(conf_file, eval_dir, model=model)
    # print(grids)
    total_required = 0
    missing_indices = list()
    for grid in grids:
        total_required += cumulative_mul(grid.shape, 1)
        missing_indices.extend(get_missing_entries(grid))
    for indices in missing_indices:
        missing_hp_sets.append(
            get_hp_set_from_ordered_dict_by_indices(conf, indices)
        )
    return missing_hp_sets, total_required


def get_all_permutations(list_to_permute):
    """returns all permutations of list with no similar elements"""
    perms = list()
    if len(list_to_permute) > 0:
        for v in list_to_permute:
            base = [v]
            l_to_permute = list(list_to_permute)
            l_to_permute.remove(v)
            ps = get_all_permutations(l_to_permute)
            for p in ps:
                perms.append(base + p)
        return perms
    else:
        return [[]]


def get_elements(init_sequence, indices):
    if isinstance(init_sequence, (list, tuple)):
        res = list()
        for idx in indices:
            res.append(init_sequence[idx])
        if isinstance(init_sequence, tuple):
            res = tuple(res)
    elif isinstance(init_sequence, str):
        res = ''
        for idx in indices:
            res += init_sequence[idx]
    elif isinstance(init_sequence, OrderedDict):
        dict_contents = list()
        for idx, pair in enumerate(init_sequence.items()):
            dict_contents.append(pair)
        dict_init = OrderedDict()
        for idx in indices:
            dict_init.append(dict_contents[idx])
        res = OrderedDict(dict_init)
    return res


def get_hp_set_from_ordered_dict_by_indices(conf, indices):
    t = list(conf.items())
    hp_set = OrderedDict()
    if len(t) != len(indices):
        raise InvalidArgumentError(
            "Number of hyper parameter has to be equal to number of indices",
            (len(conf), len(indices)),
            "'conf' and 'indices'",
            "len(conf) == len(indices)"
        )
    for idx, hp_name_and_values in zip(indices, t):
        hp_set[hp_name_and_values[0]] = hp_name_and_values[1][idx]
    return hp_set


def form_confs_from_partially_tested_params(grid, conf, all_partially_tested_param_indices, par_num):
    partially_tested_param_indices = all_partially_tested_param_indices[par_num]
    res = list()
    partially_tested_grid = grid[tuple([slice(None)]*par_num + [partially_tested_param_indices])]
    # print("(form_confs_from_partially_tested_params)partially_tested_grid:", partially_tested_grid)
    partially_tested_conf = construct(conf)
    key, value = list(partially_tested_conf.items())[par_num]
    partially_tested_conf[key] = get_elements(value, partially_tested_param_indices)
    # print("(form_confs_from_partially_tested_params)key:", key)
    # print("(form_confs_from_partially_tested_params)partially_tested_conf:", partially_tested_conf)
    for v_idx, v in enumerate(partially_tested_conf[key]):
        new_conf_tmpl = construct(partially_tested_conf)
        for k in new_conf_tmpl.keys():
            if k != key:
                new_conf_tmpl[k] = None
            else:
                new_conf_tmpl[k] = [v]
        rec_conf = construct(partially_tested_conf)
        del rec_conf[key]
        rec_grid = partially_tested_grid[tuple([slice(None)] * par_num + [v_idx])]
        # print("(form_confs_from_partially_tested_params)rec_conf:", rec_conf)
        # print("(form_confs_from_partially_tested_params)rec_grid:", rec_grid)
        rconfs = form_confs(rec_grid, rec_conf)
        for rconf in rconfs:
            new_conf = construct(new_conf_tmpl)
            for hp_name, hp_values in rconf.items():
                new_conf[hp_name] = hp_values
            res.append(new_conf)
    return res


def form_confs(grid, conf):
    # print("(form_confs)grid:", grid)
    # print("(form_confs)conf:", conf)
    res = list()
    num_param = len(conf)
    all_tested_param_indices = list()
    all_not_tested_param_indices = list()
    all_partially_tested_param_indices = list()
    there_is_partially_tested_param_values = False
    num_not_tested_combs = list()
    for par_num, (hp_name, hp_values) in enumerate(conf.items()):
        reduce_dims = tuple([i for i in range(num_param) if i != par_num])
        full_slice_sum = np.sum(grid, reduce_dims)
        if grid.ndim > 1:
            max_num_times_tested = reduce(lambda x, y: x*y, [grid.shape[i] for i in range(num_param) if i != par_num])
        else:
            max_num_times_tested = 1
        tested_param_indices = list()
        not_tested_param_indices = list()
        partially_tested_param_indices = list()
        # print("(form_confs)full_slice_sum:", full_slice_sum)
        for value_idx, ntimes_value_tested in enumerate(np.nditer(full_slice_sum)):
            if ntimes_value_tested == max_num_times_tested:
                tested_param_indices.append(value_idx)
            elif ntimes_value_tested == 0:
                not_tested_param_indices.append(value_idx)
            else:
                partially_tested_param_indices.append(value_idx)
                there_is_partially_tested_param_values = True
        all_tested_param_indices.append(tested_param_indices)
        all_not_tested_param_indices.append(not_tested_param_indices)
        all_partially_tested_param_indices.append(partially_tested_param_indices)
        num_not_tested_combs.append(len(not_tested_param_indices) * max_num_times_tested)
    # print("(form_confs)num_not_tested_combs:", num_not_tested_combs)
    max_num_not_tested_combs = max(num_not_tested_combs)
    if max_num_not_tested_combs == 0:
        if there_is_partially_tested_param_values:
            res.extend(form_confs_from_partially_tested_params(grid, conf, all_partially_tested_param_indices, 0))
    else:
        par_num = num_not_tested_combs.index(max_num_not_tested_combs)
        whole_conf = OrderedDict(conf)
        key, value = list(whole_conf.items())[par_num]
        whole_conf[key] = get_elements(value, all_not_tested_param_indices[par_num])
        # print("(form_confs)key:", key)
        # print("(form_confs)whole_conf:", whole_conf)
        res.append(whole_conf)
        if there_is_partially_tested_param_values:
            res.extend(form_confs_from_partially_tested_params(grid, conf, all_partially_tested_param_indices, par_num))
    return res


def apply_func_to_nested(nested, func, obj_types):
    if isinstance(nested, obj_types):
        if isinstance(nested, dict):
            for k, v in nested.items():
                nested[k] = apply_func_to_nested(v, func, obj_types)
        else:
            for idx, v in enumerate(nested):
                nested[idx] = apply_func_to_nested(v, func, obj_types)
        return nested
    else:
        return func(nested)


def synchronous_sort(seqs, leader_idx, lambda_func=lambda x: x):
    new_seqs = list()
    for s in zip(*sorted(zip(*seqs), key=lambda elem: lambda_func(elem[leader_idx]))):
        new_seqs.append(list(s))
    return new_seqs


def remove_empty_strings_from_list(l):
    res = list()
    for e in l:
        if len(e) > 0:
            res.append(e)
    return res


def get_substitution_tensor(tensor, substitution_way, **kwargs):
    shape = tensor.get_shape().as_list()
    if substitution_way == 'random':
        s = tf.random_uniform(
            shape,
            minval=kwargs['minval'],
            maxval=kwargs['maxval']
        )
    elif substitution_way == 'zeros':
        s = tf.zeros(
            shape
        )
    elif substitution_way == 'constant':
        s = tf.constant(kwargs['value'], shape=shape)
    else:
        print("WARNING: unknown substitution way. No substitution is performed")
        s = tensor
    return s


def vsum(var, summary_types):
    res = list()
    if 'mean' in summary_types:
        mean = tf.reduce_mean(var)
        with tf.device('/cpu:0'):
            res.append(tf.summary.scalar('mean', mean))
    else:
        mean = None
    if 'stddev' in summary_types:
        if mean is None:
            mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        with tf.device('/cpu:0'):
            res.append(tf.summary.scalar('stddev', stddev))
    if 'max' in summary_types:
        max = tf.reduce_max(var)
        with tf.device('/cpu:0'):
            res.append(tf.summary.scalar('max', max))
    if 'min' in summary_types:
        min = tf.reduce_min(var)
        with tf.device('/cpu:0'):
            res.append(tf.summary.scalar('min', min))
    if 'histogram' in summary_types:
        with tf.device('/cpu:0'):
            res.append(tf.summary.histogram('histogram', var))
    return res


def variable_summaries(var, summary_types, name_scope):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if name_scope is None:
        return vsum(var, summary_types)
    else:
        with tf.name_scope(name_scope):
            return vsum(var, summary_types)


def get_elem_from_nested(nested, keys):
    if len(keys) > 0:
        return get_elem_from_nested(nested[keys[0]], keys[1:])
    else:
        return nested


def cumulative_mul(sequence, init):
    res = construct(init)
    for a in sequence:
        res *= a
    return res


def hyperparameter_name_string(name):
    # print('(Handler._hyperparameter_name_string)name:', name)
    string = name[1]
    if name[2] is not None:
        string += '[%s]' % name[2]
    if name[3] is not None:
        string += '/' + name[3]
    return string


def sort_hps(hps):
    return sorted(hps, key=lambda x: hyperparameter_name_string(x))


def hp_name_2_hp_description(hp_name):
    description = ['UNKNOWN']
    if '/' in hp_name:
        name, other = hp_name.split('/')
        if '[' in other:
            spec, index_part = other.split('[')
        else:
            spec = other
            index_part = None
    else:
        spec = None
        if '[' in hp_name:
            name, index_part = hp_name.split('[')
        else:
            name = hp_name
            index_part = None
    if index_part is not None:
        index = int(index_part[:-1])
    else:
        index = None
    description.extend([name, index, spec])
    return tuple(description)


def check_if_hp_description_is_in_list(description, l):
    # print("(useful_functions.check_if_hp_description_is_in_list)description:", description)
    # print("(useful_functions.check_if_hp_description_is_in_list)l:", l)
    indices = list()
    for idx, e in enumerate(l):
        if e[1:] == description[1:]:
            indices.append(idx)
    num_matches = len(indices)
    if num_matches == 0:
        print("ERROR: failed to match description %s" % description)
        raise Exception
    elif num_matches > 1:
        print("ERROR: %s matches ( " + "%s" * num_matches + ") for description %s" \
              % tuple(get_elements(l, indices) + [description]))
        raise Exception
    return True, l[indices[0]]


def all_combs(list_of_lists):
    lengths = [len(l) for l in list_of_lists]
    num_combs = cumulative_mul(lengths, 1)
    combs = list()
    for comb_num in range(num_combs):
        indices = one_dim_idx_2_multidim_indices(comb_num, lengths)
        combs.append([l[idx] for l, idx in zip(list_of_lists, indices)])
    return combs


def add_stddev(data):
    x_values = list()
    y_value_sets = list()
    # print("(useful_functions.add_stddev)data:", data)
    for x, y in zip(*data):
        # print("(useful_functions.add_stddev)x:", x)
        # print("(useful_functions.add_stddev)y:", y)
        if x in x_values:
            y_value_sets[x_values.index(x)].append(y)
        else:
            x_values.append(x)
            y_value_sets.append([y])
    # print("(useful_functions.add_stddev)x_values:", x_values)
    # print("(useful_functions.add_stddev)y_value_sets:", y_value_sets)
    means = [sum(value_set) / len(value_set) for value_set in y_value_sets]
    stddevs = list()
    for value_set, mean in zip(y_value_sets, means):
        num_values = len(value_set)
        if num_values > 1:
            square_devs = [(y - mean)**2 for y in value_set]
            stddev = np.sqrt(sum(square_devs) / (num_values - 1))
        else:
            stddev = None
        stddevs.append(stddev)
    return [x_values, means, stddevs]


def get_optimizer_evaluation_results(eval_dir, hp_order, averaging_number):
    eval_dir_contents = os.listdir(eval_dir)
    try:
        eval_dir_contents.remove('hp_layout.txt')
    except ValueError:
        raise HPLayoutMissingError(eval_dir, "File 'hp_layout.txt' is missing in dir '%s'" % eval_dir)
    except:
        raise
    edc = construct(eval_dir_contents)
    for entry in edc:
        if 'launch_log' in entry:
            eval_dir_contents.remove(entry)
        if 'plot' in entry:
            eval_dir_contents.remove(entry)

    with open(os.path.join(eval_dir, 'hp_layout.txt'), 'r') as f:
        hp_save_order = f.read().split()

    hp_value_files = list()
    result_dirs = list()
    for entry in eval_dir_contents:
        if entry[-4:] == '.txt':
            hp_value_files.append(entry)
        else:
            result_dirs.append(entry)

    hp_value_files = sorted(hp_value_files, key=lambda elem: int(elem[:-4]))
    result_dirs = sorted(result_dirs, key=lambda elem: int(elem))

    pupil_names = os.listdir(os.path.join(eval_dir, result_dirs[0]))

    res_files = os.listdir(os.path.join(eval_dir, result_dirs[0], pupil_names[0]))

    result_types = list()
    regimes = list()
    for res_file in res_files:
        [res_type, regime] = res_file.split('_')
        if res_type not in result_types:
            result_types.append(res_type)
        regime = regime.split('.')[0]
        if regime not in regimes:
            regimes.append(regime)

    hp_map = list()
    # print("(plot_hp_search)hp_save_order:", hp_save_order)
    for hp in hp_order:
        try:
            hp_map.append((hp, hp_save_order.index(hp)))
        except ValueError:
            raise MissingHPError(
                hp,
                "missing hyper parameter %s in %s" % (hp, eval_dir)
            )
    if len(hp_save_order) > len(hp_order):
        raise ExtraHPError(
            hp_order,
            hp_save_order,
            "Extra hyper parameter in evaluation directory '%s'."
            "\nhp_order: %s\nhp_names(in file): %s" % (eval_dir, hp_order, hp_save_order)
        )

    hp_map = dict(hp_map)

    for_plotting = dict()
    for pupil_name in pupil_names:
        for_plotting[pupil_name] = dict()
        for res_type in result_types:
            for_plotting[pupil_name][res_type] = dict()
            for regime in regimes:
                for_plotting[pupil_name][res_type][regime] = dict()

    file_for_hp_types = os.path.join(eval_dir, hp_value_files[0])
    # print("(plot_hp_search)file_for_hp_types:", file_for_hp_types)
    with open(file_for_hp_types, 'r') as f:
        hp_types = f.read().split('\n')[1].split()

    # print("(plot_helpers.plot_hp_search)regimes:", regimes)
    for hp_value_file, result_dir in zip(hp_value_files, result_dirs):
        hp_value_file = os.path.join(eval_dir, hp_value_file)
        with open(hp_value_file, 'r') as f:
            hp_v = [convert(v, type_) for v, type_ in zip(f.read().split('\n')[0].split(), hp_types)]
        hp_values = [hp_v[hp_map[hp]] for hp in hp_order]
        # print("(useful_functions.get_evaluation_results)hp_values:", hp_values)

        fixed_hps_tuple = tuple(hp_values[:-2])
        if len(hp_values) > 1:
            line_hp_value = hp_values[-2]
        else:
            line_hp_value = None
        changing_hp_value = hp_values[-1]
        for pupil_name in pupil_names:
            for res_type in result_types:
                for regime in regimes:
                    d = for_plotting[pupil_name][res_type][regime]
                    if fixed_hps_tuple not in d:
                        d[fixed_hps_tuple] = dict()
                    d = d[fixed_hps_tuple]
                    if line_hp_value not in d:
                        d[line_hp_value] = [list(), list()]
                    r = d[line_hp_value]
                    file_name = res_type + '_' + regime + '.txt'
                    file_with_data = os.path.join(eval_dir, result_dir, pupil_name, file_name)
                    with open(file_with_data, 'r') as f:
                        t = f.read()
                    lines = t.split('\n')
                    lines = remove_empty_strings_from_list(lines)
                    s = 0
                    # print("(plot_hp_search)lines:", lines)
                    for i in range(averaging_number):
                        s += float(lines[-i].split()[-1])
                    mean = s / averaging_number
                    # print(pupil_name, res_type, regime, fixed_hps_tuple, line_hp_value, changing_hp_value)
                    r[0].append(changing_hp_value)
                    r[1].append(mean)
    for pupil_res in for_plotting.values():
        for metric_res in pupil_res.values():
            for regime_res in metric_res.values():
                for fixed_hps_res in regime_res.values():
                    for line_key, line_res in fixed_hps_res.items():
                        fixed_hps_res[line_key] = add_stddev(line_res)

    # print("(plot_helpers.plot_hp_search)for_plotting:", for_plotting)
    for_plotting = apply_func_to_nested(for_plotting, lambda x: synchronous_sort(x, 0), (dict,))
    return for_plotting


def get_pupil_evaluation_results(eval_dir, hp_order):
    eval_dir_contents = get_pupil_eval_dir_contents(eval_dir)
    # for entry in edc:
    #     print(entry)
    #     # print(eval_dir_contents)
    #     if 'launch_log' in entry or not entry.endswith('.txt'):
    #         # print(entry)
    #         eval_dir_contents.remove(entry)
    for_plotting = dict()
    # print(eval_dir_contents)
    # print("(useful_functions.get_evaluation_results)hp_order:", hp_order)
    for entry in eval_dir_contents:
        path_to_res = os.path.join(eval_dir, entry)
        dataset_name = entry[:-4]
        # print("(useful_functions.get_pupil_evaluation_results)path_to_res:", path_to_res)
        with open(path_to_res, 'r') as f:
            lines = f.read().split('\n')
        if check_if_line_is_header(lines[0]):
            metrics, hp_names = parse_header_line(lines[0])
        else:
            raise HeaderLineError(
                lines[0],
                "Can not parse header line in evaluation file %s" % path_to_res
            )
        num_hps = len(hp_names)
        for hp_name in hp_order:
            if hp_name not in hp_names:
                raise MissingHPError(
                    hp_name, "Missing hyper parameter '%s' in evaluation file '%s'" % (hp_name, path_to_res))
        if num_hps > len(hp_order):
            raise ExtraHPError(
                hp_order,
                hp_names,
                "Extra hyper parameter in evaluation file '%s'."
                "\nhp_order: %s\nhp_names(in file): %s" % (path_to_res, hp_order, hp_names)
            )
        line_hp_is_present = (len(hp_order) > 1)

        for_plotting[dataset_name] = dict([(metric, dict()) for metric in metrics])
        for idx, line in enumerate(remove_empty_strings_from_list(lines[1:])):
            if check_if_line_contains_results(line):
                hp_set = extract_hp_set(line, hp_names)
                res = extract_metrics(line, metrics)
                hp_values = [hp_set[name] for name in hp_order]
                fixed_hps = tuple(hp_values[:-2])
                if line_hp_is_present:
                    line_hp = hp_values[-2]
                else:
                    line_hp = None
                for metric, m_value in res.items():
                    d = for_plotting[dataset_name][metric]
                    if fixed_hps not in d:
                        d[fixed_hps] = dict()
                    d = d[fixed_hps]
                    if line_hp not in d:
                        d[line_hp] = [list(), list()]
                    xy = d[line_hp]
                    xy[0].append(hp_values[-1])
                    xy[1].append(m_value)
            else:
                print(
                    "WARNING: %s line '%s' in '%s' can not be parsed" % (idx, line, path_to_res)
                )
    for dataset_res in for_plotting.values():
        for metric_res in dataset_res.values():
            for fixed_hps_res in metric_res.values():
                for line_key, line_res in fixed_hps_res.items():
                    fixed_hps_res[line_key] = add_stddev(line_res)
    for_plotting = apply_func_to_nested(for_plotting, lambda x: synchronous_sort(x, 0), (dict,))
    return for_plotting


def parse_x_select(x_select_str):
    tmp = x_select_str.split('[')
    tmp2 = [el[:-1].split(',') for el in tmp[1:]]
    x_select = [
        [convert(el[0], 'float'), convert(el[1], 'float')] for el in tmp2
    ]
    return x_select


def parse_line_select(line_select_str):
    tmp = line_select_str.split(',')
    return [convert(el, 'float') for el in tmp]


def check_in_range(x, range_):
    in_range = True
    if x < range_[0] or x > range_[1]:
        in_range = False
    return in_range


def check_in_ranges(x, ranges):
    in_ranges = False
    for range_ in ranges:
        in_ranges = in_ranges or check_in_range(x, range_)
    return in_ranges


def select_by_x(line_data, x_select):
    if x_select is None:
        return line_data
    else:
        selected_x, selected_y, selected_stddev = list(), list(), list()
        for x, y, stddev in zip(*line_data):
            if check_in_ranges(x, x_select):
                selected_x.append(x)
                selected_y.append(y)
                selected_stddev.append(stddev)
        return [selected_x, selected_y, selected_stddev]


def select_for_plot(data, select):
    selected = dict()
    for fixed_hps, fixed_hp_data in data.items():
        selected_lines = dict()
        present_line_keys = list(fixed_hp_data.keys())
        if select['line_select'] is None:
            line_keys = present_line_keys
        else:
            line_keys = select['line_select']
        for line_key in line_keys:
            if line_key in present_line_keys:
                line_data = fixed_hp_data[line_key]
                if select['x_select'] is None:
                    selected_lines[line_key] = line_data
                else:
                    selected_lines[line_key] = select_by_x(line_data, select['x_select'])
        selected[fixed_hps] = selected_lines
    return selected


def broadcast_list(l, n):
    if n > 1:
        if len(l) == 1:
            return l*n
        elif len(l) != n:
            raise FailedBroadcastError(
                n,
                l,
                "Failed to broadcast list %s to length %s" % (l, n)
            )
        else:
            return l
    else:
        return l


def broadcast_many_lists(lists, n):
    res = list()
    for l in lists:
        res.append(
            broadcast_list(l, n)
        )
    return res


def split_strings_by_char(strings, char):
    res = list()
    for string in strings:
        res.append(string.split(char))
    return res


def get_res_types(for_plotting):
    all_res_types = list()
    for res in for_plotting.values():
        res_types = list(res.keys())
        all_res_types = add_missing_to_list(all_res_types, res_types)
    return all_res_types


def get_lines_by_res_type(for_plotting, inf, x_select, model):
    res_types = list(for_plotting[inf['dataset_name']].keys())
    line_data = dict()
    if model == 'pupil':
        path_tmpl = [inf['dataset_name'], None, inf['fixed_hps'], inf['line_hp']]
    elif model == 'optimizer':
        path_tmpl = [inf['pupil_name'], None, inf['regime'], inf['fixed_hps'], inf['line_hp']]
    else:
        path_tmpl = None
    for t in res_types:
        path = construct(path_tmpl)
        path[1] = t
        line = get_obj_elem_by_path(for_plotting, path)
        # print(line)
        line = select_by_x(line, x_select)
        line_data[t] = line
    return line_data


def retrieve_lines(retrieve_inf, x_select, model, averaging_number):
    for_plotting = dict()
    all_res_types = list()
    for eval_dir, inf_list in retrieve_inf.items():
        if model == 'pupil':
            fp = get_pupil_evaluation_results(eval_dir, inf_list[0]['hp_plot_order'])
        elif model == 'optimizer':
            fp = get_optimizer_evaluation_results(eval_dir, inf_list[0]['hp_plot_order'], averaging_number)
        for_plotting[eval_dir] = fp
        # print(fp)
        res_types = get_res_types(fp)
        all_res_types = add_missing_to_list(all_res_types, res_types)
    # print(all_res_types)
    lines = dict([(t, dict()) for t in all_res_types])
    for eval_dir, inf_list in retrieve_inf.items():
        for inf in inf_list:
            lines_by_res_type = get_lines_by_res_type(
                for_plotting[eval_dir],
                inf,
                x_select,
                model,
            )
            label = inf['label']
            for res_type, line in lines_by_res_type.items():
                lines[res_type][label] = line
    return lines


def isnumber(string):
    if len(string) == 0:
        return False
    for c in string:
        if c not in NUMBER_CHARS:
            return False
    return True


def normalize(normalized, norm_bearer):
    return normalized / (global_norm([normalized]) + EPSILON) * global_norm([norm_bearer])


def get_pupil_eval_dir_contents(eval_dir):
    contents = construct(os.listdir(eval_dir))
    filtered = list()
    for entry in contents:
        if 'launch_log' not in entry and entry[-4:] == '.txt':
            filtered.append(entry)
    return filtered


def get_hp_and_metric_names_from_pupil_eval_dir(eval_dir):
    filtered = get_pupil_eval_dir_contents(eval_dir)
    with open(os.path.join(eval_dir, filtered[0]), 'r') as f:
        lines = f.read().split('\n')
    lines = remove_empty_strings_from_list(lines)
    if not check_if_line_is_header(lines[0]):
        raise HeaderLineError(
            lines[0],
            "broken header line %s in file %s" % (lines[0], os.path.join(eval_dir, filtered[0]))
        )
    metric_names, hp_names = parse_header_line(lines[0])
    return hp_names, metric_names


def get_hp_names_from_optimizer_eval_dir(eval_dir):
    with open(os.path.join(eval_dir, 'hp_layout.txt'), 'r') as f:
        hp_names = f.read().split('\n')[0]
    filtered = list()
    for name in hp_names.split():
        if len(name) != 0:
            filtered.append(name)
    return filtered


def get_dataset_names_from_eval_dir(eval_dir):
    contents = get_pupil_eval_dir_contents(eval_dir)
    return [entry[:-4] for entry in contents]


def get_pupil_names_from_eval_dir(eval_dir):
    exp_description_files, _ = get_exp_description_files_and_contents(eval_dir)
    name = exp_description_files[0][:-4]
    return os.listdir(os.path.join(eval_dir, name))


def get_metric_names_and_regimes_from_optimizer_eval_dir(eval_dir):
    exp_description_files, _ = get_exp_description_files_and_contents(eval_dir)
    name = exp_description_files[0][:-4]
    pupil_name = os.listdir(os.path.join(eval_dir, name))[0]
    result_files = os.listdir(os.path.join(eval_dir, name, pupil_name))
    metric_names = list()
    regimes = list()
    for res_file in result_files:
        spl = res_file.split('_')
        metric_name = spl[0]
        regime = spl[1].split('.')[0]
        if regime not in regimes:
            regimes.append(regime)
        if metric_name not in metric_names:
            metric_names.append(metric_name)
    return metric_names, regimes


def in_interval(x, interval):
    bra = interval[0]
    cket = interval[-1]
    left, right = interval[1:-1].split(',')
    if is_float(left):
        left = float(left)
        if bra == '[':
            if x >= left:
                left_ok = True
            else:
                left_ok = False
        elif bra == '(':
            if x > left:
                left_ok = True
            else:
                left_ok = False
    else:
        left_ok = True

    if is_float(right):
        right = float(right)
        if cket == ']':
            if x <= right:
                right_ok = True
            else:
                right_ok = False
        elif cket == ')':
            if x < right:
                right_ok = True
            else:
                right_ok = False
    else:
        right_ok = True
    return left_ok and right_ok


def split_by_brackets(value_filter):
    splitted = re.split('([\]\)])', value_filter)
    length = len(splitted)
    intervals = list()
    for i in range(length // 2):
        intervals.append(
            splitted[i*2] + splitted[i*2+1]
        )
    # print(intervals)
    return intervals


def pass_filter(x, value_filter):
    if math.isnan(x):
        return False
    if math.isinf(x):
        return False
    value_filter = split_by_brackets(value_filter)
    filter_is_passed = False
    for interval in value_filter:
        filter_is_passed = filter_is_passed or in_interval(x, interval)
    return filter_is_passed


def search_hps_giving_min_and_max(data, value_filter):
    min = None
    max = None
    fixed_hps_num = len(list(data.keys())[0])
    line_hp = list(list(data.values())[0].keys())[0]
    if line_hp is None:
        max_res_hp = [None] * fixed_hps_num + [None]
        min_res_hp = [None] * fixed_hps_num + [None]
    else:
        max_res_hp = [None] * fixed_hps_num + [line_hp, None]
        min_res_hp = [None] * fixed_hps_num + [line_hp, None]
    for fixed_hps_tuple, fdata in data.items():
        for line_hp, ldata in fdata.items():
            for changing_hp, v, _ in zip(*ldata):
                if (max is None or v > max) and pass_filter(v, value_filter):
                    max = v
                    if line_hp is None:
                        max_res_hp = list(fixed_hps_tuple) + [changing_hp]
                    else:
                        max_res_hp = list(fixed_hps_tuple) + [line_hp, changing_hp]
                if (min is None or v < min) and pass_filter(v, value_filter):
                    min = v
                    if line_hp is None:
                        min_res_hp = list(fixed_hps_tuple) + [changing_hp]
                    else:
                        min_res_hp = list(fixed_hps_tuple) + [line_hp, changing_hp]
    return dict(
        min=[tuple(min_res_hp), min],
        max=[tuple(max_res_hp), max]
    )


def filter_wrapper(value_filter):
    def f(data):
        return search_hps_giving_min_and_max(data, value_filter)
    return f


def get_min_and_max(for_plotting, model, value_filter='[-inf,0)(0,+inf]'):
    if model == 'pupil':
        depth = 2
    elif model == 'optimizer':
        depth = 3
    f = filter_wrapper(value_filter)
    return apply_to_nested_on_depth(for_plotting, f, depth)


def apply_direction(minmax, metric):
    if METRIC_IMPROVEMENT_DIRECTIONS[metric] == 'up':
        return minmax['max']
    else:
        return minmax['min']


def get_best(for_plotting, model, value_filter='[-inf,0)(0,+inf]'):
    minmax = get_min_and_max(for_plotting, model, value_filter=value_filter)
    res = dict()
    if model == 'pupil':
        for dataset_name, dataset_res in minmax.items():
            res[dataset_name] = dict()
            rd = res[dataset_name]
            for metric, metric_minmax in dataset_res.items():
                rd[metric] = apply_direction(metric_minmax, metric)
    elif model == 'optimizer':
        # print("(useful_functions.get_best)pupil_names:", list(minmax.keys()))
        for pupil_name, pupil_res in minmax.items():
            res[pupil_name] = dict()
            rp = res[pupil_name]
            # print("(useful_functions.get_best)metrics:", list(pupil_res.keys()))
            for metric, metric_res in pupil_res.items():
                rp[metric] = dict()
                rm = rp[metric]
                # print("(useful_functions.get_best)regimes:", list(metric_res.keys()))
                for regime, regime_minmax in metric_res.items():
                    rm[regime] = apply_direction(regime_minmax, metric)
    return res


def print_hps(hp_names, hp_values, indent):
    indent_str = ' ' * indent
    for hp_name, hp_value in zip(hp_names, hp_values):
        print(indent_str + hp_name + ':', hp_value)


def parse_path_comb(eval_dir, filter_=True):
    dir_sets = eval_dir.split(':')
    dir_sets_prepared = list()
    for dir_set in dir_sets:
        dir_sets_prepared.append(dir_set.split(','))
    eval_dirs = [os.path.join(*comb) for comb in all_combs(dir_sets_prepared)]
    # print("(plot_helpers.parse_path_comb)eval_dirs:", eval_dirs)
    if filter_:
        filtered = list()
        for dir_ in eval_dirs:
            if os.path.exists(dir_):
                filtered.append(dir_)
            else:
                print("WARNING: %s directory does not exist" % dir_)
    else:
        filtered = eval_dirs
    # print("(plot_helpers.parse_path_comb)filtered:", filtered)
    return filtered


def get_points_from_range(string):
    [start, end, num_points, scale] = string.split(',')
    start = convert(start, 'float')
    end = convert(end, 'float')
    num_points = int(num_points)

    points = list()
    if scale == 'linear':
        step = (end - start) / num_points
        significant_digit = int(max(np.log10(abs(end / step)), np.log10(abs(start / step))))
        for i in range(num_points + 1):
            point = start + i * step
            points.append(point)
    if scale in ['log', 'symlog']:
        if start * end < 0:
            raise InvalidArgumentError(
                'if scale is log or symlog start*end should be positive',
                allowed_values='start*end>0',
                value=[start, end],
                name="start and end",
            )
        else:
            factor = (end / start) ** (1. / num_points)
            significant_digit = int(np.log10((1. / abs(abs(factor) - 1.)))) + 1

            for i in range(num_points + 1):
                point = start * factor ** i
                points.append(point)
    if significant_digit < 0:
        significant_digit = 0
    format = '{:0.%se}' % significant_digit
    if scale == 'symlog':
        for p in list(points):
            points.append(-p)
        points.append(0.)
    return [format.format(p) for p in points]


def get_tmpl(l):
    length = len(l)
    return '%s ' * (length - 1) + '%s'


def get_values(values_str):
    return [convert(v, 'float') for v in remove_empty_strings_from_list(values_str.split())]


def get_borders(borders_str):
    return [convert(v, 'float') for v in borders_str.split(',')]


def apply_borders(values, borders):
    res = list()
    for v in values:
        if borders[0] <= v <= borders[1]:
            res.append(v)
    return res


def perform_cut(old_values_str, borders_str):
    old_values = get_values(old_values_str)
    borders = get_borders(borders_str)
    new_values = apply_borders(old_values, borders)
    return new_values


def perform_sym_cut(old_values_str, borders_str):
    old_values = get_values(old_values_str)
    borders = get_borders(borders_str)
    all_borders = [
        borders,
        [-borders[1], -borders[0]]
    ]
    new_values = list()
    for bs in all_borders:
        new_values.extend(apply_borders(old_values, bs))
    if .0 not in new_values:
        new_values.append(.0)
    return new_values


def get_pupil_result_paths(model_path):
    res = dict()
    contents = os.listdir(model_path)
    for entry in contents:
        path = os.path.join(model_path, entry)
        if is_int(entry) and os.path.isdir(path):
            res[entry] = path
    return res


def get_values_in_pupil_test_results(all_results, dataset, metric):
    tmp = [[e, v[dataset][metric]] for e, v in all_results.items()]
    exps, values = zip(*tmp)
    return list(exps), list(values)


def compute_stddev(l):
    num_el = len(l)
    mean = float(sum(l)) / num_el
    sqdif = [(v-mean)**2 for v in l]
    if num_el > 1:
        return np.sqrt(sum(sqdif) / (num_el - 1))
    else:
        return None


def get_stats_of_pupil_test_results(all_results):
    res = dict()
    exps = list(all_results.keys())
    datasets = list(all_results[exps[0]].keys())
    metrics = list(all_results[exps[0]][datasets[0]].keys())
    for dataset in datasets:
        res[dataset] = dict()
        for metric in metrics:
            exps, values = get_values_in_pupil_test_results(all_results, dataset, metric)
            maxv = max(values)
            maxe = exps[values.index(maxv)]
            minv = min(values)
            mine = exps[values.index(minv)]
            num_exp = len(exps)
            mean = sum(values) / num_exp
            stddev = compute_stddev(values)
            res[dataset][metric] = dict(
                maximum=maxv,
                minimum=minv,
                stddev=stddev,
                mean=mean,
                maxexp=maxe,
                minexp=mine,
                numexp=num_exp,
            )
    return res


def pupil_test_results_summarize(model_path, inter_path):
    res_dirs = get_pupil_result_paths(model_path)
    all_results = dict()
    for exp, path in res_dirs.items():
        res = dict()
        test_dir = os.path.join(path, inter_path, 'results')
        test_result_files = os.listdir(test_dir)
        for file in test_result_files:
            path_to_file = os.path.join(test_dir, file)
            with open(path_to_file, 'r') as f:
                v = float(f.read())
            spl = file.split('_')
            metric = spl[0]
            dataset = '_'.join(spl[1:])[:-4]
            if dataset not in res:
                res[dataset] = {metric: v}
            else:
                res[dataset][metric] = v
        all_results[exp] = res
    stats = get_stats_of_pupil_test_results(all_results)
    return stats


def indent_text(text, indent):
    res = ' ' * indent
    for c in text:
        if c == '\n':
            res += c + ' ' * indent
        else:
            res += c
    return res


def stats_string(stats):
    string = ''
    if 'mean' in stats:
        string += 'mean: %s' % stats['mean'] + '\n'
    if 'stddev' in stats:
        string += 'stddev: %s' % stats['stddev'] + '\n'
    if 'maximum' in stats:
        maxstr = 'max: %s' % stats['maximum']
        if 'maxexp' in stats:
            maxstr += '(%s)' % stats['maxexp']
        string += maxstr + '\n'
    if 'minimum' in stats:
        minstr = 'min: %s' % stats['minimum']
        if 'minexp' in stats:
            minstr += '(%s)' % stats['minexp']
        string += minstr + '\n'
    if 'numexp' in stats:
        string += 'number of experiments: %s' % stats['numexp'] + '\n'
    return string[:-1]


def stats_string_for_all_metrics(metric_stats, indent):
    string = ''
    for metric, stats in metric_stats.items():
        string += metric + ':\n'
        string += indent_text(stats_string(stats), indent) + '\n'
    return string[:-1]


def stats_string_for_all_datasets(dataset_stats, indent):
    string = ''
    for dataset, stats in dataset_stats.items():
        string += dataset + ':\n'
        string += indent_text(stats_string_for_all_metrics(stats, indent), indent) + '\n'
    return string[:-1]


def stats_string_for_all_models(model_stats, indent):
    string = ''
    for model, stats in model_stats.items():
        string += model + ':\n'
        string += indent_text(stats_string_for_all_datasets(stats, indent), indent) + '\n'
    return string[:-1]


def get_transform_func(tr):
    if tr == 'identity':
        return lambda x: x
    elif tr == '1_minus_x':
        return lambda x: 1. - x
    else:
        return None


def perform_transformation(old_file, new_eval_dir, old_names, new_names, types, transformations):
    path = os.path.split(old_file)
    new_file = os.path.join(os.path.split(path[0])[0], new_eval_dir, path[-1])
    create_path(new_file, file_name_is_in_path=True)
    with open(old_file, 'r') as f:
        lines = remove_empty_strings_from_list((f.read().split('\n')))
        header_line = lines[0]
        res_lines = lines[1:]
    metrics, names = parse_header_line(header_line)
    names = metrics + names
    transforms = list()
    for_header = list()
    for n in names:
        if n in old_names:
            idx = old_names.index(n)
            for_header.append(new_names[idx])
            transforms.append(get_transform_func(transformations[idx]))
        else:
            for_header.append(n)
            transforms.append(get_transform_func('identity'))
    new_header_line = ' '.join(for_header)
    num_params = len(names)
    tmpl = '%s ' * (num_params - 1) + '%s'
    if types is None:
        types = ['float'] * num_params
    with open(new_file, 'w') as f:
        f.write(new_header_line + '\n')
        for l in res_lines:
            new_values = list()
            for idx, (v, t) in enumerate(zip(l.split(), types)):
                new_values.append(
                    transforms[idx](convert(v, t))
                )
            f.write(tmpl % tuple(new_values) + '\n')


def metric_word_index(words):
    indices = list()
    for idx, w in enumerate(words):
        if w in METRICS:
            indices.append(idx)
    return indices[-1]


def parse_line_file_name(file_name):
    words = file_name.split('_')
    words[-1] = words[-1][:-4]
    last_metric_word_idx = metric_word_index(words)
    metric_name = '_'.join(words[:last_metric_word_idx+1])
    dataset_name = '_'.join(words[last_metric_word_idx+1:])
    return metric_name, dataset_name


def parse_1_line_dir(dir):
    contents = os.listdir(dir)
    res = dict()
    for file_name in contents:
        metric, dataset_name = parse_line_file_name(file_name)
        if metric not in res:
            res[metric] = {
                dataset_name: os.path.join(dir, file_name)
            }
        else:
            res[metric][dataset_name] = os.path.join(dir, file_name)
    return res


def keys_from_list_in_dict(dictionary, keys):
    res = list()
    for key in keys:
        if key in dictionary:
            res.append(key)
    return res


def extract_line_from_file(file_name):
    with open(file_name, 'r') as f:
        lines = remove_empty_strings_from_list(f.read().split('\n'))
    num_entries = len(lines[0].split())
    data = [list() for _ in range(num_entries)]
    for line in lines:
        values = [float(v) for v in line.split()]
        for v, l in zip(values, data):
            l.append(v)
    if num_entries == 2:
        data.append([None] * len(data[0]))
    return data


def form_combinations_from_dicts(
        dictionaries
):
    maps = [dict() for _ in dictionaries]
    varying_values = list()
    varying_counter = 0
    for d, map_ in zip(dictionaries, maps):
        for name, values in d.items():
            varying_values.append(values)
            map_[name] = varying_counter
            varying_counter += 1
    all_combinations = all_combs(varying_values)
    all_insertions = list()
    for comb in all_combinations:
        one_set_of_of_insertions = [dict() for _ in dictionaries]
        for one_type_insertions, map_ in zip(one_set_of_of_insertions, maps):
            for name, idx in map_.items():
                one_type_insertions[name] = comb[idx]
        all_insertions.append(one_set_of_of_insertions)
    return all_insertions


def unite_dicts_with_conflicting_keys(list_of_dicts):
    united = dict()
    for idx, d in enumerate(list_of_dicts):
        for k, v in d.items():
            united[(idx, k)] = v
    return united


def sort_lines(lines):
    return apply_to_nested_on_depth(
        lines,
        lambda x: synchronous_sort(x, 0),
        obj_types=(dict,),
    )


def extend_for_relative(data):
    if len(data) > 0:
        if len(data[0]) > 2:
            relative_provided = True
        else:
            relative_provided = False
    else:
        relative_provided = False
    if not relative_provided:
        return data
    res = list()
    for point in data:
        specs = construct(point[0])
        specs.append(
            dict(
                unit='seconds'
            )
        )
        res.append([specs, point[1]])
        specs = construct(point[0])
        specs.append(
            dict(
                unit='relative'
            )
        )
        res.append([specs, point[2]])
    return res


def transform_data_into_dictionary_of_lines(data, order):
    depth = len(order)
    if depth > 1:
        res = dict()
    else:
        res = [list(), list()]
    for point in data:
        v = point[1]
        specs = unite_dicts_with_conflicting_keys(point[0])
        d = res
        for idx, name in enumerate(order):
            if idx < depth - 2:
                if specs[name] not in d:
                    d[specs[name]] = dict()
                d = d[specs[name]]
            elif idx == depth - 2:
                if specs[name] not in d:
                    d[specs[name]] = [list(), list()]
                d = d[specs[name]]
            else:
                if specs[name] not in d[0]:
                    d[0].append(specs[name])
                    d[1].append(v)
    return sort_lines(res)


def optimizer_time_measurement_save_order(names, base):
    order = list()
    for name in names:
        order.append((1, name))
    if base is not None:
        order.insert(-2, (3, 'unit'))
    return order


def save_lines(lines, dir_):
    if isinstance(lines, dict):
        for k, v in lines.items():
            path = os.path.join(dir_, str(k))
            save_lines(v, path)
    else:
        file_name = os.path.join(dir_ + '.txt')
        create_path(file_name, file_name_is_in_path=True)
        with open(file_name, 'w') as f:
            num_points = len(lines[0])
            for p_idx, (x, y) in enumerate(zip(*lines)):
                if isinstance(x, list):
                    x = x[0]
                if isinstance(y, list):
                    y = y[0]
                f.write('%s %s' % (x, y))
                if p_idx < num_points - 1:
                    f.write('\n')


def shift_list(l, sh):
    ll = list()
    for v in l:
        ll.append(v+sh)
    return ll
