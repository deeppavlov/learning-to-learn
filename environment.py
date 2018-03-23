import time
import numpy as np
import re
import queue
import csv
import sys
import select
import multiprocessing as mp
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from useful_functions import InvalidArgumentError
from useful_functions import (construct, add_index_to_filename_if_needed, match_two_dicts, create_path,
                              check_if_key_in_nested_dict, add_missing_to_list, print_and_log,
                              apply_temperature, sample, is_int)
from args_parsing import parse_1_set_of_kwargs, parse_train_method_arguments, \
    formalize_and_create_insertions_for_build_hps, formalize_and_create_insertions_for_other_hps, \
    create_all_args_for_launches, configure_args_for_launches
from handler import Handler
from subword_nmt.apply_bpe import BPE
from bpe import prepare_for_bpe, bpe_post_processing

class Controller(object):
    """Controller is a class which instances are used for computing changing learning parameters. For example
    learning rate. It is also responsible for training stopping
    Usage:
        1. Construct controller by passing him 'storage' (a dictionary with monitored parameters, usually if used in
        _train method of Environment class 'storage' is self._storage) and specifications. Specifications is
        a dictionary with necessary parameters for computing the controlled value
        2. Use 'get' method to get current value
        3. If you wish to use Controller instance with new specifications you should:
            - add new private method to Controller which will be responsible for processing new specifications. It
              should take no arguments and return monitored value.
            - add elif entry in __init__ method for assigning 'get' with newly created method
            - if new approach requires parameters not provided in self._storage than add them. Also don't forget
              to pass this parameters to _update_storage method in the bottom of _train"""
    @staticmethod
    def create_change_tracking_specifications(specifications):
        if isinstance(specifications, list):
            old_specs = construct(specifications)
        if isinstance(specifications, dict):
            old_specs = [dict(specifications)]
        new_specs = dict()
        new_specs['old_specs'] = old_specs
        new_specs['type'] = 'changes_detector'
        return new_specs

    def __init__(self, storage, specifications):
        self._storage = storage
        self._specifications = specifications
        if self._specifications['type'] == 'limit_steps':
            self.get = self._limit_steps
        elif self._specifications['type'] == 'exponential_decay':
            self.get = self._exponential_decay
        elif self._specifications['type'] == 'fixed':
            self.get = self._fixed
        elif self._specifications['type'] == 'periodic_truth':
            self.get = self._periodic_truth
        elif self._specifications['type'] == 'true_on_steps':
            self.get = self._true_on_steps
        elif self._specifications['type'] == 'always_false':
            self.get = self._always_false
        elif self._specifications['type'] == 'changes_detector':
            self._value_controllers = list()
            self._last_values = list()
            for value_specs in self._specifications['old_specs']:
                self._value_controllers.append(Controller(self._storage, value_specs))
                self._last_values.append(self._value_controllers[-1].get())
            self.get = self._changes_detector

        elif self._specifications['type'] == 'linear':
            self.get = self._linear

    def _changes_detector(self):
        something_changed = False
        for idx, (last_value, controller) in enumerate(zip(self._last_values, self._value_controllers)):
            if last_value != controller.get():
                something_changed = something_changed or True
                self._last_values[idx] = controller.get()
        return something_changed

    def _exponential_decay(self):
        num_stairs = self._storage['step'] // self._specifications['period']
        returned_value = self._specifications['init']
        return returned_value * self._specifications['decay']**num_stairs

    def _linear(self):
        start = self._specifications['start']
        end = self._specifications['end']
        step_interval = self._specifications['interval']
        step = self._storage['step']
        if step < step_interval:
            return start + (end - start) * step / step_interval
        else:
            return end

    def _limit_steps(self):
        if self._storage['step'] > self._specifications['limit']:
            return False
        else:
            return True

    def _fixed(self):
        return self._specifications['value']

    def _periodic_truth(self):
        if self._storage['step'] % self._specifications['period'] == 0:
            return True
        else:
            return False

    def _true_on_steps(self):
        if self._storage['step'] in self._specifications['steps']:
            return True
        else:
            return False

    @staticmethod
    def _always_false():
        return False

    @property
    def name(self):
        return self._specifications['name']


def perplexity_tensor(**kwargs):
    with tf.name_scope('computing_perplexity'):
        probabilities = kwargs['probabilities']
        labels = kwargs['labels']
        special_args = kwargs['special_args']
        probabilities_shape = probabilities.get_shape().as_list()
        length = probabilities_shape[1]
        if 'dialog_switch' in special_args:
            if special_args['dialog_switch']:
                _, switch = tf.split(labels, [length, 1], axis=1, name='switch_vector')
                switch = tf.reshape(switch, [-1], name='switch_reshaped')
        if 'mark_vec_len' in special_args:
            if special_args['mark_vec_len'] is not None:
                probabilities, _ = tf.split(
                    probabilities,
                    [length - special_args['mark_vec_len'], special_args['mark_vec_len']],
                    axis=1,
                    name='word_and_punctuation_preds')
        ln2 = np.log(2, dtype=np.float32)
        shape = probabilities.get_shape().as_list()
        probabilities = tf.where(probabilities > 1e-10,
                                 probabilities,
                                 np.full(tuple(shape), 1e-10),
                                 name='to_small_values_in_probs_are_filtered')
        log_probabilities = tf.divide(tf.log(probabilities), ln2, name='log2_probs')
        entropy = tf.reduce_sum(- probabilities * log_probabilities, axis=1, name='entropy_not_mean')
        perplexity = tf.exp(ln2 * entropy, name='perplexity_not_aver')
        if 'dialog_switch' in special_args:
            if special_args['dialog_switch']:
                perplexity = tf.multiply(perplexity, switch, name='relevant_perplexity')
                num_of_sensible_results = tf.reduce_sum(switch, name='num_of_sensible_results')
                there_is_sensible = tf.not_equal(num_of_sensible_results, 0., name='there_is_sensible')
                mean_perplexity = tf.divide(tf.reduce_sum(perplexity, name='sum_perplexity'),
                                            (num_of_sensible_results + 1e-12),
                                            name='mean_perplexity')
                return tf.where(there_is_sensible, mean_perplexity, -1., name='perplexity')
        return tf.reduce_mean(perplexity, name="perplexity")


def loss_tensor(**kwargs):
    with tf.name_scope('computing_loss'):
        predictions = kwargs['predictions']
        labels = kwargs['labels']
        special_args = kwargs['special_args']
        predictions_shape = predictions.get_shape().as_list()
        length = predictions_shape[1]
        # print('(loss_tensor)length:', length)
        if 'dialog_switch' in special_args:
            if special_args['dialog_switch']:
                labels, switch = tf.split(labels, [length, 1], axis=1, name='labels_and_switch')
                switch = tf.reshape(switch, [-1], name='switch_reshaped')
        if 'mark_vec_len' in special_args:
            if special_args['mark_vec_len'] is not None:
                predictions, _ = tf.split(
                    predictions,
                    [length - special_args['mark_vec_len'], special_args['mark_vec_len']],
                    axis=1,
                    name='word_and_punctuation_preds')
                labels, _ = tf.split(
                    labels,
                    [length - special_args['mark_vec_len'], special_args['mark_vec_len']],
                    axis=1,
                    name='word_and_punctuation_labels')
        shape = predictions.get_shape().as_list()
        predictions = tf.where(predictions > 1e-10,
                               predictions,
                               np.full(tuple(shape), 1e-10),
                               name='to_small_values_in_probs_are_filtered')
        log_predictions = tf.log(predictions, name='log_pred')
        loss_on_characters = tf.reduce_sum(-labels * log_predictions, axis=1, name='loss_not_mean')
        if 'dialog_switch' in special_args:
            if special_args['dialog_switch']:
                loss_on_characters = tf.multiply(loss_on_characters, switch, name='relevant_loss')
                num_of_sensible_results = tf.reduce_sum(switch, name='num_of_sensible_results')
                there_is_sensible = tf.not_equal(num_of_sensible_results, 0., name='there_is_sensible')
                mean_loss = tf.divide(tf.reduce_sum(loss_on_characters, name='loss_sum'),
                                      (num_of_sensible_results + 1e-12),
                                      name='mean_loss')
                return tf.where(there_is_sensible, mean_loss, -1., name='loss')
        return tf.reduce_mean(loss_on_characters, name='loss')


def bpc_tensor(**kwargs):
    with tf.name_scope('computing_bpc'):
        return kwargs['loss'] / np.log(2)


def accuracy_tensor(**kwargs):
    with tf.name_scope('computing_accuracy'):
        predictions = kwargs['predictions']
        labels = kwargs['labels']
        special_args = kwargs['special_args']
        predictions_shape = predictions.get_shape().as_list()
        length = predictions_shape[1]
        if 'dialog_switch' in special_args:
            if special_args['dialog_switch']:
                labels, switch = tf.split(labels,
                                          [length, 1],
                                          axis=1,
                                          name='labels_and_switch')
                switch = tf.reshape(switch, [-1], name='switch')
        if 'mark_vec_len' in special_args:
            if special_args['mark_vec_len'] is not None:
                predictions, _ = tf.split(
                    predictions,
                    [length - special_args['mark_vec_len'], special_args['mark_vec_len']],
                    axis=1,
                    name='word_and_punctuation_preds')
                labels, _ = tf.split(
                    labels,
                    [length - special_args['mark_vec_len'], special_args['mark_vec_len']],
                    axis=1,
                    name='word_and_punctuation_labels')
        predictions = tf.argmax(predictions, axis=1, name='predictions')
        labels = tf.argmax(labels, axis=1, name='labels')

        # predictions = tf.Print(
        #     predictions,
        #     [predictions],
        #     message='predictions_in_accuracy:', summarize=1200)
        # labels = tf.Print(labels, [labels], message='labels_in_accuracy:', summarize=1200)

        accuracy = tf.to_float(tf.equal(predictions, labels), name='accuracy_not_aver')
        if 'dialog_switch' in special_args:
            if special_args['dialog_switch']:
                accuracy = tf.multiply(accuracy, switch, name='accuracy_relevant')
                num_of_sensible_results = tf.reduce_sum(switch, name='num_of_sensible_results')
                # num_of_sensible_results = tf.Print(num_of_sensible_results,
                #                                    [num_of_sensible_results],
                #                                    message='Number of sensible results in accuracy:')
                there_is_sensible = tf.not_equal(num_of_sensible_results, 0., name='there_is_sensible')
                mean_accuracy = tf.divide(tf.reduce_sum(accuracy, name='accuracy_sum'),
                                          (num_of_sensible_results + 1e-12),
                                          name='mean_accuracy')
                return tf.where(there_is_sensible, mean_accuracy, -1., name='accuracy')
        return tf.reduce_mean(accuracy, name='accuracy')


def identity_tensor(**kwargs):
    if len(kwargs) > 1:
        raise InvalidArgumentError('kwargs should not contain 1 entry', kwargs, 'kwargs', 'len(kwargs)=1')
    for value in kwargs.values():
        return value


class Environment(object):

    @staticmethod
    def put_result_types_in_correct_order(result_types):
        correct_order = ['loss', 'perplexity', 'accuracy', 'bpc']
        sorted_types = list()
        for result_type in correct_order:
            if result_type in result_types:
                sorted_types.append(result_type)
        return sorted_types

    def __init__(self,
                 pupil_class=None,
                 batch_generator_classes=None,
                 vocabulary=None,
                 datasets=None,
                 filenames=None,
                 texts=None,
                 meta_optimizer_class=None):
        """ Initializes environment class
        Args:
            pupil_class: is a class to which pupil model belongs
            meta_optimizer_class: is a class to which meta_optimizer model belongs if it is provided
            data_filenames: contains paths to a files with data for model training, validation and testing
                has to be a dictionary in which keys are names of datasets, values are strings with paths to files
            batch_generator_classes: """

        self._pupil_class = pupil_class
        self._pupil_type = self._pupil_class.get_name()
        self._meta_optimizer_class = meta_optimizer_class

        if datasets is not None:
            self._datasets = dict()
            for dataset in datasets:
                self._datasets[dataset[1]] = dataset
        else:
            self.datasets = dict()

        self._vocabulary = vocabulary

        if filenames is not None:
            for filename in filenames:
                key, value = self._process_dataset_filename(filename)
                self._datasets[key] = [value, key]

        if texts is not None:
            for text in texts:
                key, value = self._process_input_text_dataset(text)
                self._datasets[key] = [value, key]

        if not isinstance(batch_generator_classes, dict):
            self._batch_generator_classes = {'default': batch_generator_classes}
        else:
            self._batch_generator_classes = batch_generator_classes

        # # Just initializing attributes containing arguments for model building
        # self._pupil_building_parameters = self._pupil_class.get_building_parameters()
        # if self._meta_optimizer_class is not None:
        #     self._meta_optimizer_building_parameters = self._meta_optimizer_class.get_building_parameters()

        # An attributes containing instance of self._model_class. While graph is not built self._model is None
        self._pupil = None
        self._meta_optimizer = None

        # An attribute holding tensors which could be run. It has the form of dictionary which keys are user specified
        # descriptors of tensors and are tensors themselves
        self._hooks = dict()

        # List containing fuses. They are used for testing the model. You may feed them to the model and see how it
        # continues generating after that
        self._fuses = list()

        # An attribute holding session. Default value when there is no active sessions is None
        self._session = None

        self._build_functions = {'identity': identity_tensor}

        tensor_schedule = {'train_print_tensors': dict(),
                           'train_save_tensors': dict(),
                           'train_print_text_tensors': dict(),
                           'train_save_text_tensors': dict()}

        valid_tensor_schedule = {'valid_print_tensors': dict(),
                                 'valid_save_text_tensors': dict()}

        fuse_tensors = {'fuse_print_tensors': dict(), 'fuse_save_tensors': dict()}
        example_tensors = {'example_print_tensors': dict(), 'example_save_tensors': dict()}

        # Every results_collect_interval-th step BPC, accuracy, perplexity are collected
        # Every print_per_collected-th point containing BPC, accuracy and perplexity is printed
        # Together with every example_per_print-th point example is printed
        default_collected_while_training = {'results_collect_interval': 100,
                                            'print_per_collected': 1,
                                            'example_per_print': 1}

        default_collected_on_validation = {}

        default_learning_rate_control = {'init': 0.002,
                                         'decay': 0.8,
                                         'period': 1000,
                                         'type': 'exponential_decay',
                                         'name': 'learning_rate'}

        if len(self.datasets) > 0:
            default_dataset = self.datasets[0]
        else:
            default_dataset = None
        _, gens = zip(*sorted(self._batch_generator_classes.items()))
        self._default_batch_generator = gens[0]
        # additions_to_feed_dict have following format
        # It is a dictionary which keys are 'placeholder' and 'value'
        # 'placeholder' points to tensor alias and 'value' points to Controller specs
        # When providing additions_to_feed_dict to train method abbreviation of 'value' entry is allowed
        # if tensor does not change during learning. It is possible to pass tensor value in 'value' entry.
        self._default_train_method_args = dict(
            session_specs={'allow_soft_placement': False,
                           'gpu_memory': None,
                           'allow_growth': False,
                           'log_device_placement': False,
                           'visible_device_list': ""},
            start_specs={'restore_path': None,
                         'with_meta_optimizer': False,
                         'meta_optimizer_restore_path': None,
                         'save_path': None,
                         'result_types': self.put_result_types_in_correct_order(
                             ['loss', 'perplexity', 'accuracy']),
                         'summary': False,
                         'add_graph_to_summary': False,
                         'batch_generator_class': self._default_batch_generator,
                         'vocabulary': self._vocabulary},
            run=dict(
                train_specs={'learning_rate': construct(default_learning_rate_control),
                             'additions_to_feed_dict': list(),
                             'stop': {'type': 'limit_steps', 'limit': 10000, 'name': 'stop'},
                             'train_dataset': default_dataset,
                             'batch_size': {'type': 'fixed', 'value': 64, 'name': 'batch_size'},
                             'train_batch_kwargs': dict(),
                             'checkpoint_steps': None,
                             'debug': None,
                             'validation_datasets': None,
                             'validation_additions_to_feed_dict': list(),
                             'validation_batch_size': 1,
                             'valid_batch_kwargs': dict(),
                             'validate_tokens_by_chars': False,
                             'no_validation': False},
                schedule={'to_be_collected_while_training': construct(default_collected_while_training),
                          'printed_result_types':  self.put_result_types_in_correct_order(
                             ['loss']),
                          'printed_controllers': ['learning_rate'],
                          'fuses': None,
                          'fuse_tensors': construct(fuse_tensors),
                          # 'prediction_examples': None,
                          'example_length': None,
                          'example_tensors': construct(example_tensors),
                          'replicas': None,
                          'random': {'number_of_runs': 5,
                                     'length': 80},
                          'train_tensor_schedule': construct(tensor_schedule),
                          'validation_tensor_schedule': construct(valid_tensor_schedule)}
                    )
                                               )
        self._default_test_method_args = dict(
            session_specs={'allow_soft_placement': False,
                           'gpu_memory': None,
                           'allow_growth': False,
                           'log_device_placement': False,
                           'visible_device_list': ""},
            start_specs={'restore_path': None,
                         'save_path': None,
                         'print_results': True,
                         'result_types': self.put_result_types_in_correct_order(
                             ['loss', 'perplexity', 'accuracy']),
                         'verbose': True,
                         'batch_generator_class': self._default_batch_generator,
                         'vocabulary': self._vocabulary},
            work=dict(additions_to_feed_dict=list(),
                      debug=None,
                      validation_datasets=None,
                      validation_batch_size=1,
                      validate_tokens_by_chars=False,
                      valid_batch_kwargs=dict(),
                      printed_result_types=self.put_result_types_in_correct_order(['loss']),
                      fuses=None,
                      fuse_tensors=construct(fuse_tensors),
                      fuse_file_name=None,
                      example_length=None,
                      example_tensors=construct(example_tensors),
                      replicas=None,
                      random={'number_of_runs': 5,
                              'length': 80},
                      validation_tensor_schedule=construct(valid_tensor_schedule)
                    )
                                               )
        # This attribute is used solely for controlling learning parameters (learning rate, additions_to_feed_dict)
        # It is used by instances of Controller class
        # BPI stands for bits per input. It is cross entropy computed using logarithm for base 2
        self._handler = None
        self._storage = {'step': None}
        self._collected_result = None
        self.current_pupil_build_parameters = None
        self.current_pupil_launch_parameters = None
        self.current_optimizer_build_parameters = None
        self.current_optimizer_launch_parameters = None
        self.mp_debug_flag = 0

    def build_pupil(self, **kwargs):
        """A method building the graph
        Args:
            kwargs: key word arguments passed to self._model_class constructor
            :type kwargs: dictionary"""

        # checking if passed required arguments
        self._build_pupil(kwargs)

    def _build_pupil(self, kwargs):
        self._pupil_class.check_kwargs(**kwargs)
        self.current_pupil_build_parameters = kwargs
        # Building the graph
        self._pupil = self._pupil_class(**kwargs)

        # getting default hooks
        default_hooks = self._pupil.get_default_hooks()
        self._hooks.update(default_hooks)
        self._register_default_builders()

    def build_optimizer(self, **kwargs):
        self._meta_optimizer_class.check_kwargs(**kwargs)
        self.current_pupil_build_parameters = kwargs
        self._meta_optimizer = self._meta_optimizer_class(self._pupil, **kwargs)
        default_hooks = self._meta_optimizer.get_default_hooks()
        self._hooks.update(default_hooks)

    @staticmethod
    def _split_to_loss_and_not_loss_names(names):
        loss_names = list()
        not_loss_names = list()
        for name in names:
            if 'loss' in name:
                loss_names.append(name)
            else:
                not_loss_names.append(name)
        return loss_names, not_loss_names

    def _arguments_for_new_tensor_building(self, hooks, tensor_names):
        arguments = dict()
        for key, value in hooks.items():
            if value not in self._hooks:
                stars = '\n**********\n'
                self._hooks[value] = tf.placeholder(tf.float32)
                msg = "Warning! Adding to hooks shapeless placeholder %s " \
                      "of type tf.float32 with alias '%s'" % (self._hooks[value].name, value)
                print(stars + msg + stars)
            arguments[key] = self._hooks[value]
        for key, value in tensor_names.items():
            arguments[key] = tf.get_default_graph().get_tensor_by_name(value)
        return arguments

    def _add_hook(self, builder_name):
        if builder_name in self._builders:
            builder = self._builders[builder_name]
            kwargs = self._arguments_for_new_tensor_building(builder['hooks'],
                                                             builder['tensor_names'])
            kwargs['special_args'] = builder['special_args']
            new_tensor = builder['f'](**kwargs)
            self._hooks[builder['output_hook_name']] = new_tensor
        else:
            stars = '\n**********\n'
            msg = "Warning! Adding to hooks shapeless placeholder of type tf.float32 with alias '%s'" % builder_name
            print(stars + msg + stars)
            self._hooks[builder_name] = tf.placeholder(tf.float32, name=builder_name)

    def add_hooks(self, builder_names_or_builders=None, tensor_names=None):
        if builder_names_or_builders is None:
            builder_names_or_builders = list()
        if tensor_names is None:
            tensor_names = list()
        actual_names = list()
        for builder_name in builder_names_or_builders:
            if isinstance(builder_name, dict):
                self.register_builder(**builder_name)
                actual_names.append(builder_name['output_hook_name'])
            else:
                actual_names.append(builder_name)
        loss_builder_names, not_loss_builder_names = self._split_to_loss_and_not_loss_names(actual_names)
        #print('loss_builder_names:', loss_builder_names)
        #print('not_loss_builder_names:', not_loss_builder_names)
        for builder_name in loss_builder_names:
            self._add_hook(builder_name)
        for builder_name in not_loss_builder_names:
            self._add_hook(builder_name)
        for alias, name in tensor_names:
            self._hooks[alias] = tf.get_default_graph().get_tensor_by_name(name)

    def register_build_function(self, function, name):
        self._build_functions[name] = function

    def print_available_builders(self):
        for builder_name, builder in self._builders.items():
            print(builder_name + ':', builder)

    def register_builder(self,
                         f=None,
                         hooks=None,
                         tensor_names=None,
                         output_hook_name=None,
                         special_args=None):
        if hooks is None:
            hooks = dict()
        if tensor_names is None:
            tensor_names = dict()
        if isinstance(f, str):
            f = self._build_functions[f]
        self._builders[output_hook_name] = dict(f=f,
                                                hooks=hooks,
                                                tensor_names=tensor_names,
                                                output_hook_name=output_hook_name,
                                                special_args=special_args)

    def _register_default_builders(self):
        pupil_special_args = self._pupil.get_special_args()
        train_perplexity_builder = dict(f=perplexity_tensor,
                                        hooks={'probabilities': 'predictions',
                                               'labels': 'labels_prepared'},
                                        tensor_names=dict(),
                                        output_hook_name='perplexity',
                                        special_args=pupil_special_args)
        valid_perplexity_builder = dict(f=perplexity_tensor,
                                        hooks={'probabilities': 'validation_predictions',
                                               'labels': 'validation_labels_prepared'},
                                        tensor_names=dict(),
                                        output_hook_name='validation_perplexity',
                                        special_args=pupil_special_args)
        valid_loss_builder = dict(f=loss_tensor,
                                  hooks={'predictions': 'validation_predictions',
                                         'labels': 'validation_labels_prepared'},
                                  tensor_names=dict(),
                                  output_hook_name='validation_loss',
                                  special_args=pupil_special_args)
        train_bpc_builder = dict(f=bpc_tensor,
                                 hooks={'loss': 'loss'},
                                 tensor_names=dict(),
                                 output_hook_name='bpc',
                                 special_args=pupil_special_args)
        valid_bpc_builder=dict(f=bpc_tensor,
                               hooks={'loss': 'validation_loss'},
                               tensor_names=dict(),
                               output_hook_name='validation_bpc',
                               special_args=pupil_special_args)
        train_accuracy_builder=dict(f=accuracy_tensor,
                                    hooks={'predictions': 'predictions',
                                          'labels': 'labels_prepared'},
                                    tensor_names=dict(),
                                    output_hook_name='accuracy',
                                    special_args=pupil_special_args)
        valid_accuracy_builder=dict(f=accuracy_tensor,
                                    hooks={'predictions': 'validation_predictions',
                                           'labels': 'validation_labels_prepared'},
                                    tensor_names=dict(),
                                    output_hook_name='validation_accuracy',
                                    special_args=pupil_special_args)

        self._builders = {'perplexity': train_perplexity_builder,
                          'validation_perplexity': valid_perplexity_builder,
                          'validation_loss': valid_loss_builder,
                          'bpc': train_bpc_builder,
                          'validation_bpc': valid_bpc_builder,
                          'accuracy': train_accuracy_builder,
                          'validation_accuracy': valid_accuracy_builder}

    @classmethod
    def _update_dict(cls, dict_to_update, update):
        """Checks if update matches dict_to_update and updates it
        Args:
            dict_to_update: a class attribute of type dict which should be updated
            update: dict which is used for updating"""
        keys_all_right = match_two_dicts(update, dict_to_update)
        if keys_all_right:
            for key, value in update.items():
                if isinstance(value, dict):
                    cls._update_dict(dict_to_update[key], update[key])
                else:
                    dict_to_update[key] = construct(value)

    @property
    def default_train_method_args(self):
        return construct(self._default_train_method_args)

    @default_train_method_args.setter
    def default_train_method_args(self, update):
        """update is a dictionary which should match keys of self._pupil_default_training"""
        self._update_dict(self._default_train_method_args, update)

    @property
    def default_test_method_args(self):
        return construct(self._default_test_method_args)

    @default_test_method_args.setter
    def default_test_method_args(self, update):
        """update is a dictionary which should match keys of self._pupil_default_training"""
        self._update_dict(self._default_test_method_args, update)

    def get_default_method_parameters(self,
                                      method_name):
        if method_name == 'train':
            return self.default_train_method_args
        if method_name == 'test':
            return self.default_test_method_args
        return None

    def _start_session(self, allow_soft_placement, log_device_placement, gpu_memory, allow_growth, visible_device_list):
        """Starts new session with specified parameters. If there is opend session closes it"""
        if self._session is not None:
            print('Warning: there is an opened session already. Closing it')
            self._session.close()
        # print('(_start_session)gpu_memory:', gpu_memory)
        # print('(_start_session)allow_growth:', allow_growth)
        config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory,
                                                          allow_growth=allow_growth,
                                                          visible_device_list=visible_device_list),
                                log_device_placement=log_device_placement
                                )
        # config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
        self._session = tf.Session(config=config)

    def _close_session(self):
        self._session.close()
        self._session = None

    def init_storage(self, dataset_name, **kwargs):
        self._storage[dataset_name] = dict()
        d = self._storage[dataset_name]
        for key, value in kwargs.items():
            d[key] = value

    def append_to_storage(self, dataset_name, **kwargs):
        if dataset_name is not None:
            d = self._storage[dataset_name]
        else:
            d = self._storage
        for key, value in kwargs.items():
            d[key].append(value)

    def flush_storage(self):
        self._storage = {'step': None}

    def set_in_storage(self, **kwargs):
        for key, value in kwargs.items():
            self._storage[key] = value

    def check_if_key_in_storage(self, keys):
        return check_if_key_in_nested_dict(self._storage, keys)

    def _create_checkpoint(self, step, checkpoints_path, model_type='pupil'):
        path = checkpoints_path + '/' + str(step)
        print('\nCreating checkpoint at %s' % path)
        if model_type == 'pupil':
            self._hooks['saver'].save(self._session, path)
        elif model_type == 'meta_optimizer':
            self._hooks['saver'].save(self._session, path)

    def _restore_pupil(self, restore_path):
        if restore_path is not None:
            print('restoring pupil from %s' % restore_path)
            self._hooks['saver'].restore(self._session, restore_path)

    def _restore_meta_optimizer(self, restore_path):
        if restore_path is not None:
            print('restoring meta optimizer from %s' % restore_path)
            self._hooks['meta_optimizer_saver'].restore(self._session, restore_path)

    def test(self,
             **kwargs):
        self.flush_storage()
        self._store_launch_parameters('pupil', **kwargs)
        tmp_output = parse_1_set_of_kwargs(self,
                                           kwargs,
                                           'test',
                                           None,
                                           False)
        all_tensor_aliases = self._all_tensor_aliases_from_test_method_arguments(tmp_output)
        # print('all_tensor_aliases:', all_tensor_aliases)
        self._create_missing_hooks(all_tensor_aliases)
        session_specs = tmp_output['session_specs']
        start_specs = tmp_output['start_specs']
        work = tmp_output['work']
        dataset_names = [dataset[1] for dataset in work['validation_datasets']]
        # print("work['fuses']:", work['fuses'])
        self._start_session(session_specs['allow_soft_placement'],
                            session_specs['log_device_placement'],
                            session_specs['gpu_memory'],
                            session_specs['allow_growth'],
                            session_specs['visible_device_list'])
        self._session.run(tf.global_variables_initializer())
        self._restore_pupil(start_specs['restore_path'])
        add_feed_dict = dict()
        # print("(Environment.test)work['additions_to_feed_dict']:", work['additions_to_feed_dict'])
        for addition in work['additions_to_feed_dict']:
            add_feed_dict[self._hooks[addition['placeholder']]] = addition['value']
        batch_generator_class = start_specs['batch_generator_class']
        self._handler = Handler(self,
                                self._hooks,
                                'test',
                                start_specs['save_path'],
                                start_specs['result_types'],
                                save_to_file=True,
                                save_to_storage=True,
                                print_results=start_specs['print_results'],
                                batch_generator_class=batch_generator_class,
                                vocabulary=start_specs['vocabulary'],
                                validation_dataset_names=dataset_names,
                                validation_tensor_schedule=work['validation_tensor_schedule'],
                                fuses=work['fuses'],
                                fuse_tensor_schedule=work['fuse_tensors'],
                                fuse_file_name=work['fuse_file_name'],
                                verbose=start_specs['verbose'])
        # print('(Environment.test)self._storage:', self._storage)
        self._handler.log_launch()
        empty_batch_gen = batch_generator_class('', 1, vocabulary=start_specs['vocabulary'])
        if work['fuses'] is not None:
            fuse_res = self._on_fuses(empty_batch_gen,
                                      work['fuses'],
                                      additional_feed_dict=add_feed_dict)
        else:
            fuse_res = None

        validation_datasets = work['validation_datasets']
        # print("(Environment.test)work['valid_batch_kwargs']:", work['valid_batch_kwargs'])
        for validation_dataset in validation_datasets:
            if work['validate_tokens_by_chars']:
                _ = self._validate_by_chars(
                    batch_generator_class, validation_dataset, work['validation_batch_size'],
                    work['valid_batch_kwargs'], additional_feed_dict=add_feed_dict)
            else:
                # print('(Environment.test)self._storage:', self._storage)
                _ = self._validate(
                    batch_generator_class, validation_dataset, work['validation_batch_size'],
                    work['valid_batch_kwargs'], additional_feed_dict=add_feed_dict)
        if work['example_length'] is not None:
            example_res = list()
            for validation_dataset in validation_datasets:
                # print('(Environment.test)self._storage:', self._storage)
                example_res.append(
                    self._prediction_examples(
                        batch_generator_class,
                        validation_dataset,
                        work['example_length'],
                        work['valid_batch_kwargs'],
                        additional_feed_dict=add_feed_dict
                    )
                )
        else:
            example_res = None
        self._close_session()
        return fuse_res, example_res

    def _on_fuses(self,
                  batch_generator,
                  fuses,
                  training_step=None,
                  additional_feed_dict=None):
        if additional_feed_dict is None:
            additional_feed_dict = dict()
        for fuse_idx, fuse in enumerate(fuses):
            if fuse_idx % 100 == 0:
                print('Number of processed fuses:', fuse_idx)
            self._handler.set_processed_fuse_index(fuse_idx)
            for repeat_idx in range(fuse['num_repeats']):
                if 'randomize_sample_state' in self._hooks:
                    self._session.run(self._hooks['randomize_sample_state'])
                elif 'reset_validation_state' in self._hooks:
                    self._session.run(self._hooks['reset_validation_state'])
                # print("fuse['text']:", [fuse['text']])
                for char_idx, char in enumerate(fuse['text']):
                    vec = batch_generator.char2vec(char, batch_generator.character_positions_in_vocabulary)
                    feed_dict = {self._hooks['validation_inputs']: vec}
                    feed_dict.update(additional_feed_dict)
                    fuse_operations = self._handler.get_tensors('fuse', char_idx)
                    # print('(_on_fuses)feed_dict:', feed_dict)
                    fuse_res = self._session.run(fuse_operations, feed_dict=feed_dict)
                    if char_idx == len(fuse['text']) - 1 and fuse['max_num_of_chars'] > 0:
                        self._handler.start_fuse_accumulation()
                    self._handler.process_results(char_idx, fuse_res, regime='fuse')
                # self._handler.start_fuse_accumulation()
                if fuse['fuse_stop'] == 'limit':
                    for char_idx in range(len(fuse['text']), len(fuse['text']) + fuse['max_num_of_chars'] - 1):
                        vec = batch_generator.pred2vec(fuse_res[0])
                        feed_dict = {self._hooks['validation_inputs']: vec}
                        feed_dict.update(additional_feed_dict)
                        fuse_operations = self._handler.get_tensors('fuse', char_idx)
                        fuse_res = self._session.run(fuse_operations, feed_dict=feed_dict)
                        self._handler.process_results(char_idx, fuse_res, regime='fuse')
                elif fuse['fuse_stop'] == 'new_line':
                    char = None
                    counter = 0
                    char_idx = len(fuse['text'])
                    while char != '\n' and counter < fuse['max_num_of_chars'] - 1:
                        vec = batch_generator.pred2vec(fuse_res[0])
                        feed_dict = {self._hooks['validation_inputs']: vec}
                        feed_dict.update(additional_feed_dict)
                        fuse_operations = self._handler.get_tensors('fuse', char_idx)
                        fuse_res = self._session.run(fuse_operations, feed_dict=feed_dict)
                        self._handler.process_results(char_idx, fuse_res, regime='fuse')
                        char = batch_generator.vec2char(fuse_res[0], batch_generator.vocabulary)[0]
                        # print('char:', char)
                        counter += 1
                        char_idx += 1
                self._handler.stop_fuse_accumulation()
            self._handler.set_processed_fuse_index(None)
        res = self._handler.dispense_fuse_results(training_step)
        return res

    def _prediction_examples(self,
                             batch_generator_class,
                             validation_dataset,
                             example_length,
                             valid_batch_kwargs,
                             additional_feed_dict=None,
                             training_step=None):
        if additional_feed_dict is None:
            additional_feed_dict = dict()
        example_batches = batch_generator_class(validation_dataset[0], 1, **valid_batch_kwargs)
        self._handler.start_example_accumulation()
        for c_idx in range(min(example_length, example_batches.get_dataset_length()) + 1):
            inputs, _ = example_batches.next()
            input_str = batch_generator_class.vec2char_fast(
                np.reshape(inputs, (1, -1)),
                self._vocabulary)[0]
            # print('(Environment._prediction_examples)inputs:', inputs)
            # print('(Environment._prediction_examples)input_str:', input_str)
            feed_dict = {self._hooks['validation_inputs']: inputs}
            feed_dict.update(additional_feed_dict)
            example_operations = self._handler.get_tensors('example', c_idx)
            # print('(_prediction_examples)feed_dict:', feed_dict)
            example_res = self._session.run(example_operations, feed_dict=feed_dict)
            self._handler.process_results(c_idx, input_str, example_res, regime='example')
        self._handler.stop_example_accumulation()
        res = self._handler.dispense_example_results(training_step)
        return res

    def _validate(self,
                  batch_generator_class,
                  validation_dataset,
                  validation_batch_size,
                  valid_batch_kwargs,
                  training_step=None,
                  additional_feed_dict=dict(),
                  save_to_file=None,
                  save_to_storage=None,
                  print_results=None):
        # print('valid_batch_kwargs:', valid_batch_kwargs)
        if 'reset_validation_state' in self._hooks:
            self._session.run(self._hooks['reset_validation_state'])
        #print('batch_generator_class:', batch_generator_class)
        valid_batches = batch_generator_class(validation_dataset[0], validation_batch_size, **valid_batch_kwargs)
        length = valid_batches.get_dataset_length()
        inputs, labels = valid_batches.next()
        step = 0
        self._handler.start_accumulation(validation_dataset[1], training_step=training_step)
        while step < length:
            validation_operations = self._handler.get_tensors('validation', step)
            feed_dict = {self._hooks['validation_inputs']: inputs,
                         self._hooks['validation_labels']: labels}
            if isinstance(additional_feed_dict, dict):
                feed_dict.update(additional_feed_dict)
            valid_res = self._session.run(validation_operations, feed_dict=feed_dict)
            self._handler.process_results(training_step, valid_res, regime='validation')
            step += 1
            inputs, labels = valid_batches.next()

        means = self._handler.stop_accumulation(save_to_file=save_to_file,
                                                save_to_storage=save_to_storage,
                                                print_results=print_results)
        return means

    def _validate_by_chars(
            self,
            batch_generator_class,
            validation_dataset,
            validation_batch_size,
            valid_batch_kwargs,
            training_step=None,
            additional_feed_dict=None,
            save_to_file=None,
            save_to_storage=None,
            print_results=None):
        if additional_feed_dict is None:
            additional_feed_dict = dict()
        # print('valid_batch_kwargs:', valid_batch_kwargs)
        if 'reset_validation_state' in self._hooks:
            self._session.run(self._hooks['reset_validation_state'])
        #print('batch_generator_class:', batch_generator_class)
        valid_batches = batch_generator_class(validation_dataset[0], validation_batch_size, **valid_batch_kwargs)
        length = valid_batches.get_dataset_length()
        inputs, labels, correct_tokens = valid_batches.next_with_tokens()
        step = 0
        self._handler.start_accumulation(validation_dataset[1], training_step=training_step)
        while step < length:
            validation_operations = self._handler.get_tensors('validation', step)
            feed_dict = {self._hooks['validation_inputs']: inputs,
                         self._hooks['validation_labels']: labels}
            if isinstance(additional_feed_dict, dict):
                feed_dict.update(additional_feed_dict)
            valid_res = self._session.run(validation_operations, feed_dict=feed_dict)
            self._handler.process_results(training_step, valid_res, correct_tokens[0], regime='validation_by_chars')
            step += 1
            inputs, labels, correct_tokens = valid_batches.next_with_tokens()

        means = self._handler.stop_accumulation(save_to_file=save_to_file,
                                                save_to_storage=save_to_storage,
                                                print_results=print_results)
        return means

    def _from_random_fuse(self):
        pass

    def _on_replicas(self):
        pass

    def _get_all_tensors_from_schedule(self, schedule):
        returned_list = list()
        for _, dict_with_tensors in schedule.items():
            for tensor_alias in dict_with_tensors.keys():
                returned_list.append(tensor_alias)
        return returned_list

    def _check_if_validation_is_needed(self, run_specs_set):
        """This method is not finished yet. Fuses, random and replicas should also be taken in account"""
        validation_is_needed = False
        for run_specs in run_specs_set:
            validation_is_needed = validation_is_needed or \
                                   not run_specs['train_specs']['no_validation']
        return validation_is_needed

    def _all_tensor_aliases_from_train_method_arguments(self, args_for_launches, evaluation=None):
        start_specs_for_launches, run_specs_for_launches = zip(*args_for_launches)
        list_of_required_tensors_aliases = list()
        result_types_for_launches = list()
        for start_specs in start_specs_for_launches:
            if start_specs['with_meta_optimizer']:
                result_types_for_launches = add_missing_to_list(
                    result_types_for_launches, start_specs['result_types'])
        list_of_required_tensors_aliases.extend(result_types_for_launches)
        for start_specs, run_specs_set in zip(start_specs_for_launches, run_specs_for_launches):
            if start_specs['with_meta_optimizer']:
                if self._check_if_validation_is_needed(run_specs_set):
                    for result_type in start_specs['result_types']:
                        list_of_required_tensors_aliases.append('validation_' + result_type)

        for run_specs_set in run_specs_for_launches:
            for run_specs in run_specs_set:
                train_aliases = self._get_all_tensors_from_schedule(run_specs['schedule']['train_tensor_schedule'])
                list_of_required_tensors_aliases = add_missing_to_list(list_of_required_tensors_aliases, train_aliases)
                valid_aliases = self._get_all_tensors_from_schedule(run_specs['schedule']['validation_tensor_schedule'])
                list_of_required_tensors_aliases = add_missing_to_list(list_of_required_tensors_aliases, valid_aliases)
        if evaluation is not None:
            if 'train' in evaluation['datasets'] and len(evaluation['datasets']) > 1:
                for result_type in evaluation['result_types']:
                    alias = 'validation_' + result_type
                    if alias not in list_of_required_tensors_aliases:
                        list_of_required_tensors_aliases.append(alias)
        return list_of_required_tensors_aliases

    @staticmethod
    def _tensor_aliases_from_schedule(schedule):
        tensor_aliases = list()
        for _, schedule in schedule.items():
            aliases = list(schedule.keys())
            tensor_aliases = add_missing_to_list(tensor_aliases, aliases)
        return tensor_aliases

    def _all_tensor_aliases_from_test_method_arguments(self, args):
        start_specs = args['start_specs']
        work = args['work']
        list_of_required_tensors_aliases = list()
        for res_type in start_specs['result_types']:
            list_of_required_tensors_aliases.append('validation_' + res_type)
        list_of_required_tensors_aliases = add_missing_to_list(
            list_of_required_tensors_aliases,
            self._tensor_aliases_from_schedule(work['fuse_tensors']))
        list_of_required_tensors_aliases = add_missing_to_list(
            list_of_required_tensors_aliases,
            self._tensor_aliases_from_schedule(work['validation_tensor_schedule']))
        return list_of_required_tensors_aliases

    def _create_missing_hooks(self, list_of_tensor_aliases):
        missing = list()
        for tensor_alias in list_of_tensor_aliases:
            if tensor_alias not in self._hooks:
                missing.append(tensor_alias)
        self.add_hooks(missing)

    def _build_batch_kwargs(self, unprepared_kwargs):
        kwargs = dict()
        for key, arg in unprepared_kwargs.items():
            if isinstance(arg, Controller):
                kwargs[key] = arg.get()
            else:
                kwargs[key] = arg
        return kwargs

    def _form_validation_additional_feed_dict(self,
                                              train_feed_dict_additions,
                                              additional_controllers,
                                              validation_additional_feed_dict):

        valid_add_feed_dict = dict()
        for addition, add_controller in zip(train_feed_dict_additions, additional_controllers):
            valid_add_feed_dict[self._hooks[addition['placeholder']]] = add_controller.get()
        for addition in validation_additional_feed_dict:
            valid_add_feed_dict[self._hooks[addition['placeholder']]] = addition['value']
        return valid_add_feed_dict

    def _train(self,
               run_specs,
               checkpoints_path,
               batch_generator_class,
               with_meta_optimizer,
               init_step=0):
        """It is a method that does actual training and responsible for one training pass through dataset. It is called
        from train method (maybe several times)
        Args:
            kwargs should include all entries defined in self._pupil_default_training"""
        train_specs = construct(run_specs['train_specs'])
        schedule = construct(run_specs['schedule'])
        step = init_step

        # creating batch generator

        # resetting step in control_storage
        self.set_in_storage(step=step)
        if not with_meta_optimizer:
            learning_rate_controller = Controller(self._storage,
                                                  train_specs['learning_rate'])
        train_feed_dict_additions = train_specs['additions_to_feed_dict']
        validation_additional_feed_dict = train_specs['validation_additions_to_feed_dict']

            # print('train_feed_dict_additions:', train_feed_dict_additions)
        additional_controllers = list()
        for addition in train_feed_dict_additions:
            additional_controllers.append(Controller(self._storage, addition['value']))

        if train_specs['stop']['type'] == 'limit_steps':
            train_specs['stop']['limit'] += init_step
        should_continue = Controller(self._storage, train_specs['stop'])

        to_be_collected_while_training = schedule['to_be_collected_while_training']
        collect_interval = to_be_collected_while_training['results_collect_interval']
        print_per_collected = to_be_collected_while_training['print_per_collected']
        example_per_print = to_be_collected_while_training['example_per_print']

        if train_specs['no_validation'] or collect_interval is None:
            it_is_time_for_validation = Controller(self._storage,
                                                   {'type': 'always_false'})
            it_is_time_for_example = Controller(self._storage,
                                                {'type': 'always_false'})
        else:
            valid_period = collect_interval * print_per_collected
            it_is_time_for_validation = Controller(self._storage,
                                                   {'type': 'periodic_truth',
                                                    'period': valid_period})
            if example_per_print is None:
                it_is_time_for_example = Controller(self._storage,
                                                    {'type': 'always_false'})
            else:
                example_period = valid_period * example_per_print
                it_is_time_for_example = Controller(self._storage,
                                                    {'type': 'periodic_truth',
                                                     'period': example_period})

        if train_specs['checkpoint_steps'] is not None and checkpoints_path is not None:
            if train_specs['checkpoint_steps']['type'] == 'true_on_steps':
                for idx in range(len(train_specs['checkpoint_steps']['steps'])):
                    train_specs['checkpoint_steps']['steps'][idx] += init_step
            it_is_time_to_create_checkpoint = Controller(self._storage, train_specs['checkpoint_steps'])
        else:
            it_is_time_to_create_checkpoint = Controller(self._storage,
                                                         {'type': 'always_false'})

        batch_size_controller = Controller(self._storage, train_specs['batch_size'])
        batch_size_change_tracker_specs = Controller.create_change_tracking_specifications(train_specs['batch_size'])
        batch_size_should_change = Controller(self._storage, batch_size_change_tracker_specs)

        if train_specs['debug'] is not None:
            should_start_debugging = Controller(self._storage, train_specs['debug'])
        else:
            should_start_debugging = Controller(self._storage,
                                                {'type': 'true_on_steps',
                                                 'steps': []})

        train_batch_kwargs = dict()
        train_batch_kwargs_controller_specs = list()
        for key, batch_arg in train_specs['train_batch_kwargs'].items():
            if isinstance(batch_arg, dict):
                if 'type' in batch_arg:
                    train_batch_kwargs[key] = Controller(self._storage, batch_arg)
                    train_batch_kwargs_controller_specs.append(batch_arg)
                else:
                    train_batch_kwargs[key] = batch_arg
            else:
                train_batch_kwargs[key] = batch_arg
        change_tracker_specs = Controller.create_change_tracking_specifications(
            train_batch_kwargs_controller_specs)
        batch_generator_specs_should_change = Controller(self._storage, change_tracker_specs)

        if not with_meta_optimizer:
            controllers_for_printing = [learning_rate_controller]
        else:
            controllers_for_printing = list()

        controllers_for_printing.extend(additional_controllers)
        controllers_for_printing.append(batch_size_controller)
        batch_kwargs_controllers = list()
        for batch_kwarg in train_batch_kwargs.values():
            if isinstance(batch_kwarg, Controller):
                batch_kwargs_controllers.append(batch_kwarg)
        controllers_for_printing.extend(batch_kwargs_controllers)
        self._handler.set_new_run_schedule(schedule,
                                           train_specs['train_dataset'][1],
                                           [dataset[1] for dataset in train_specs['validation_datasets']])
        self._handler.set_controllers(controllers_for_printing)

        batch_size = batch_size_controller.get()
        tb_kwargs = self._build_batch_kwargs(train_batch_kwargs)
        train_batches = batch_generator_class(train_specs['train_dataset'][0], batch_size, **tb_kwargs)
        feed_dict = dict()
        while should_continue.get():
            if should_start_debugging.get():
                self._session = tf_debug.LocalCLIDebugWrapperSession(self._session)
                self._session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            if batch_size_should_change.get():
                batch_size = batch_size_controller.get()
                train_batches.change_batch_size(batch_size)

            if batch_generator_specs_should_change.get():
                tb_kwargs = self._build_batch_kwargs(train_batch_kwargs)
                train_batches.change_specs(**tb_kwargs)

            if it_is_time_to_create_checkpoint.get():
                self._create_checkpoint(step, checkpoints_path)
            train_inputs, train_labels = train_batches.next()

            if not with_meta_optimizer:
                learning_rate = learning_rate_controller.get()
                feed_dict[self._hooks['learning_rate']] = learning_rate

            if isinstance(self._hooks['inputs'], list):
                for input_tensor, input_value in zip(self._hooks['inputs'], train_inputs):
                    feed_dict[input_tensor] = input_value
            else:
                feed_dict[self._hooks['inputs']] = train_inputs
            if isinstance(self._hooks['labels'], list):
                for label_tensor, label_value in zip(self._hooks['labels'], train_labels):
                    feed_dict[label_tensor] = label_value
            else:
                feed_dict[self._hooks['labels']] = train_labels
            for addition, add_controller in zip(train_feed_dict_additions, additional_controllers):
                feed_dict[self._hooks[addition['placeholder']]] = add_controller.get()
            # print('(Environment._train)self._hooks:', self._hooks)
            train_operations = self._handler.get_tensors('train', step, with_meta_optimizer=with_meta_optimizer)
            # print('train_operations:', train_operations)
            # print('feed_dict:', feed_dict)

            train_res = self._session.run(train_operations, feed_dict=feed_dict)
            # here loss is given in bits per input (BPI)

            self._handler.process_results(step, train_res, regime='train')
            if it_is_time_for_validation.get():
                if len(train_specs['validation_datasets']) > 0:
                    valid_add_feed_dict = self._form_validation_additional_feed_dict(train_feed_dict_additions,
                                                                                     additional_controllers,
                                                                                     validation_additional_feed_dict)
                for validation_dataset in train_specs['validation_datasets']:
                    if train_specs['validate_tokens_by_chars']:
                        # print('(Environment._train)ready to validate by chars')
                        _ = self._validate_by_chars(
                            batch_generator_class, validation_dataset, train_specs['validation_batch_size'],
                            train_specs['valid_batch_kwargs'], training_step=step,
                            additional_feed_dict=valid_add_feed_dict)
                    else:
                        _ = self._validate(
                            batch_generator_class, validation_dataset, train_specs['validation_batch_size'],
                            train_specs['valid_batch_kwargs'], training_step=step,
                            additional_feed_dict=valid_add_feed_dict)
            if it_is_time_for_example.get():
                valid_add_feed_dict = self._form_validation_additional_feed_dict(train_feed_dict_additions,
                                                                                 additional_controllers,
                                                                                 validation_additional_feed_dict)
                if schedule['fuses'] is not None:
                    _ = self._on_fuses(train_batches,
                                       schedule['fuses'],
                                       training_step=step,
                                       additional_feed_dict=valid_add_feed_dict)
                for validation_dataset in train_specs['validation_datasets']:
                    if schedule['example_length'] is not None:
                        _ = self._prediction_examples(
                            batch_generator_class,
                            validation_dataset,
                            schedule['example_length'],
                            train_specs['valid_batch_kwargs'],
                            training_step=step,
                            additional_feed_dict=valid_add_feed_dict)
            step += 1
            self.set_in_storage(step=step)
        return step

    def train(self,
              *args,
              start_session=True,
              close_session=True,
              set_passed_parameters_as_default=False,
              **kwargs):
        """The method responsible for model training. User may specify what intermediate results he wishes to
        collect. He may regulate learning process (see arguments). It is also possible to start learning from a check
        point. User may choose if he wishes to limit number of steps
        Args:
            args: A list of arbitrary number of dictionaries which entries are similar to structure of kwargs. It is
                used if user wishes to train model consequently on several datasets. If any dictionary in list contains
                less entries than the previous one, missing entries are taken from previous. If the first doesn't have
                all entries missing entries are filled with default values
            start_session: shows if new session should be created or already opened should be used
            close_session: shows if session should be closed at the end of training
            set_passed_parameters_as_default: if True parameters of launch are saved to self._pupil_default_training.
                If args are provided the first args[0] is used for self._pupil_default_training resetting
            kwargs:
                This argument specifies the learning should be performed. There are many options and it is not
                necessary to provide all kwargs - missing will be filled with default values specified in
                _default_train_method_args atribute
                allow_soft_placement: if True tensorflow is allowed to override device assignments specified by user and
                    put ops on available devices
                gpu_memory: memory fraction tensorflow allowed to allocate. If None all available memory is allocated
                log_device_placement: If True device placements are printed to console
                restore_path: If provided graph will be restored from checkpoint
                save_path: path to directory where all results are saved
                result_types: specifies what types of results should be collected. loss, perplexity, accuracy, bpc are
                    available
                summary: If True summary writing is activated
                add_graph_to_summary: If True graph is added to summary
                batch_generator_class: class of batch generator. It has to have certain methods for correct functioning
                meta_optimizer: If meta learning is used for model training it is name of meta_optimizer network
                learning_rate: specifications for learning_rate control. If it is a float learning rate will not change
                    while learning. Otherwise it should be a dictionary. Now only exponential decay option is availbale.
                    Below dictionary entries are described
                    exponential decay:
                        type: str 'exponential_decay'
                        init: float, initial learning rate
                        decay: a factor on which learning rate is multiplied every period of steps
                        period: number of steps after which learning rate is being decreased
                additions_to_feed_dict: If your model requires some special placeholders filling (e. g. probability
                    distribution for a stochastic node) it is provided through additions_to_feed_dict. It is a
                    dictionary which keys are tensor aliases in _pupil_hooks attribute and values are dictionaries
                    of the same structure as learning_rate
                stop: specifies when learning should be stopped. It is either an integer (number of steps after which
                    learning is being stopped) or a dictionary of the same structure as learning_rate where you may
                    specify custom way of learning interruption
                train_dataset: A dataset on which model will be trained. It can be a name of dataset provided earlier to
                    Environment constructor or just something what you wish to pass to batch generator (file name, str,
                    etc.)
                batch_size: integer or dictionary of the same type as learning_rate if you wish to somehow change batch
                    size during learning
                train_batch_kwargs: If your batch generator requires some specific arguments they can be provided
                    through this dictionary (for example num_unrollings). This dictionary is used for batch generator
                    construction for training (any of batch generator parameters can be provided as key word args
                    separately if their processing is described in _process_batch_kwargs_shortcut method. Now it is only
                    'vocabulary' and 'num_unrollings')
                checkpoint_steps: list of steps on which checpoints should be created
                debug: step on which tfdbg should be activated. Default is None
                validation_dataset_names: list of dataset names used for validation (datasets have to provided to
                    Environment instance separately. Now only through constructor
                validation_dataset_texts: list of texts (type str) used for validation
                validation_dataset_filenames: file names of datasets used for validation
                  (if validation_dataset_names, validation_dataset_texts, validation_dataset_filenames provided together
                   all of them are used)
                validation_batch_size: batch size for validation
                valid_batch_kwargs: same as train_batch_kwargs
                to_be_collected_while_training: a dictionary with 3 entries (all of them can be provided independently)
                    results_collect_interval: number of steps after which data is collected
                    print_per_collected: every print_per_collected-th point collected with results_collect_interval
                        schedule is printed
                    example_per_print: every example_per_print print examples of model functioning are printed
                        (continuing from random letter, from specified fuse, responding on user specified replicas)
                printed_result_types: what model should print. Default is loss. perplexity, accuracy, bpc are also
                    available
                printed_controllers: if during learning some hyperparameters are changing you may print them to
                    console. Default printed is learning rate
                fuses: specifies fuses from which model should periodically generate text. This option is not
                    available yet
                fuse_tensors: tensor aliases from _pupil_hooks attribute which should be either saved or printed.
                    not available
                replicas: If dialog agent is trained it can be tested with consequently feeding it with few user
                    specified replicas. It can be used to check if agent is capable of dialog context accumulating
                random: NLP agents can be tested on text generating task. It is provided with first character and
                    then tries to generate text. This argument is responsible for specifying how many times it will
                    be performed and specifying length of generated sequences (not available)
                train_tensor_schedule: If user wishes he may print or save any tensor in the graph (not available)
                valid_tensor_schedule: same as train_tensor_schedule"""
        self._store_launch_parameters(
            'pupil',
            args=args,
            start_session=start_session,
            close_session=close_session,
            set_passed_parameters_as_default=set_passed_parameters_as_default,
            kwargs=kwargs)
        tmp_output = parse_train_method_arguments(self,
                                                  args,
                                                  kwargs,
                                                  set_passed_parameters_as_default=set_passed_parameters_as_default)
        session_specs = tmp_output['session_specs']
        start_specs = tmp_output['start_specs']
        run_specs_set = tmp_output['run']
        all_tensor_aliases = self._all_tensor_aliases_from_train_method_arguments([(start_specs, run_specs_set)])
        self._create_missing_hooks(all_tensor_aliases)

        if start_session:
            self._start_session(session_specs['allow_soft_placement'],
                                session_specs['log_device_placement'],
                                session_specs['gpu_memory'],
                                session_specs['allow_growth'],
                                session_specs['visible_device_list'])
        self._train_repeatedly(start_specs, run_specs_set)
        if close_session:
            self._close_session()

    def _train_repeatedly(self, start_specs, run_specs_set):
        # initializing model
        self.flush_storage()
        self._session.run(tf.global_variables_initializer())
        self._restore_pupil(start_specs['restore_path'])
        if start_specs['with_meta_optimizer']:
            self._restore_meta_optimizer(start_specs['meta_optimizer_restore_path'])

        # print('start_specs:', start_specs)
        if start_specs['with_meta_optimizer']:
            processing_type = 'train_with_meta'
        else:
            processing_type = 'train'

        self._handler = Handler(self,
                                self._hooks,
                                processing_type,
                                start_specs['save_path'],
                                start_specs['result_types'],
                                summary=start_specs['summary'],
                                add_graph_to_summary=start_specs['add_graph_to_summary'],
                                batch_generator_class=start_specs['batch_generator_class'],
                                vocabulary=start_specs['vocabulary'])
        self._handler.log_launch()
        if start_specs['save_path'] is not None:
            checkpoints_path = start_specs['save_path'] + '/checkpoints'
            create_path(checkpoints_path)
        else:
            checkpoints_path = None
        init_step = 0
        for run_specs in run_specs_set:
            init_step = self._train(run_specs,
                                    checkpoints_path,
                                    start_specs['batch_generator_class'],
                                    start_specs['with_meta_optimizer'],
                                    init_step=init_step)
        if checkpoints_path is not None:
            self._create_checkpoint('final', checkpoints_path)
        self._handler.log_finish_time()
        self._handler.close()

    def _several_launches_without_rebuilding(self,
                                             queue,
                                             kwargs_for_building,
                                             session_specs,
                                             args_for_launches,
                                             evaluation):

        self._build_pupil(kwargs_for_building)
        #print('args_for_launches:', args_for_launches)
        all_tensor_aliases = self._all_tensor_aliases_from_train_method_arguments(
            args_for_launches, evaluation=evaluation)
        #print('all_tensor_aliases:', all_tensor_aliases)
        self._create_missing_hooks(all_tensor_aliases)
        self._start_session(session_specs['allow_soft_placement'],
                            session_specs['log_device_placement'],
                            session_specs['gpu_memory'],
                            session_specs['allow_growth'],
                            session_specs['visible_device_list'])
        datasets = dict(evaluation['datasets'].items())
        if 'train' in datasets:
            del datasets['train']
        if evaluation['batch_gen_class'] is None:
            eval_batch_gen_class = self._default_batch_generator
        else:
            eval_batch_gen_class = evaluation['batch_gen_class']

        additional_feed_dict = self._form_validation_additional_feed_dict([], [], evaluation['additional_feed_dict'])
        for start_specs, run_specs_set in args_for_launches:
            result = dict()
            self._train_repeatedly(start_specs, run_specs_set)
            if 'train' in evaluation['datasets']:
                tr_res = dict()
                for key, res in self._storage['train'].items():
                    if len(res) > 0:
                        tr_res[key] = res[-1]
                result['train'] = tr_res
            self._handler = Handler(self,
                                    self._hooks,
                                    'test',
                                    None,
                                    evaluation['result_types'])
            for dataset_name, dataset in datasets.items():
                #print('dataset_name:', dataset_name)
                #print('dataset:', dataset)
                means = self._validate(eval_batch_gen_class,
                                       dataset,
                                       evaluation['batch_size'],
                                       evaluation['batch_kwargs'],
                                       additional_feed_dict=additional_feed_dict,
                                       save_to_file=False,
                                       save_to_storage=False,
                                       print_results=False)
                result[dataset_name] = means
            #print('result in process:', result)
            queue.put(result)

    @staticmethod
    def _check_hp_in_additional_feed_dict(additions, tensor_alias):
        for addition_idx, addition in enumerate(additions):
            if addition['placeholder'] == tensor_alias:
                return addition_idx
        return None

    @staticmethod
    def _check_if_controller_specs_match(default_specs, new_specs):
        if default_specs is None:
            return False
        if not isinstance(default_specs, dict):
            return False
        for key, value in new_specs.items():
            if key not in default_specs:
                return False
            if value is not None:
                if default_specs[key] != new_specs[key]:
                    return False
        return True

    def grid_search(self,
                    evaluation,
                    kwargs_for_building,
                    build_hyperparameters=None,
                    other_hyperparameters=None,
                    **kwargs):
        if build_hyperparameters is None:
            build_hyperparameters = dict()
        if other_hyperparameters is None:
            other_hyperparameters = dict()
        self._store_launch_parameters(
            'pupil',
            evaluation=evaluation,
            kwargs_for_building=kwargs_for_building,
            build_hyperparameters=build_hyperparameters,
            other_hyperparameters=other_hyperparameters,
            kwargs=kwargs)
        tmp_output = parse_train_method_arguments(self,
                                                  [],
                                                  kwargs,
                                                  set_passed_parameters_as_default=False)
        session_specs = tmp_output['session_specs']

        build_hp_combs, build_insertions = formalize_and_create_insertions_for_build_hps(build_hyperparameters)
        other_hp_combs, other_insertions = formalize_and_create_insertions_for_other_hps(other_hyperparameters)
        # print('Environment.grid_search')
        # print('build_hp_combs:', build_hp_combs)
        # print('build_insertions:', build_insertions)
        # print('other_hp_combs:', other_hp_combs)
        # print('other_insertions:', other_insertions)

        args_for_launches = create_all_args_for_launches(kwargs, other_insertions)

        hps = list()
        if len(build_hp_combs) > 0:
            hps.extend(list(build_hp_combs[0].keys()))
        if len(other_hp_combs) > 0:
            hps.extend(list(other_hp_combs[0].keys()))
        self._handler = Handler(self,
                                self._hooks,
                                'several_launches',
                                evaluation['save_path'],
                                evaluation['result_types'],
                                eval_dataset_names=list(evaluation['datasets'].keys()),
                                hyperparameters=hps)
        self._handler.log_launch()
        # print('build_insertions:', build_insertions)
        # print('build_hp_combs:', build_hp_combs)
        if len(build_hp_combs) > 0:
            for one_set_of_insertions_and_shares, build_hp_comb in zip(build_insertions, build_hp_combs):
                # print('one_set_of_insertions_and_shares:', one_set_of_insertions_and_shares)
                # print('build_hp_comb:', build_hp_comb)
                only_build_insertions = list()
                shares = list()
                for insertion, share in one_set_of_insertions_and_shares:
                    only_build_insertions.append(insertion)
                    shares.append(share)
                build_kwargs = self._pupil_class.form_kwargs(construct(kwargs_for_building),
                                                             only_build_insertions)
                parsed = configure_args_for_launches(self, args_for_launches, shares)
                queue = mp.Queue()
                # from some_useful_functions import nested2string
                # print('build_kwargs:', nested2string(build_kwargs))
                # print('parsed:', nested2string(parsed))
                self.mp_debug_flag += 1
                p = mp.Process(target=self._several_launches_without_rebuilding,
                               args=(queue, build_kwargs, session_specs, parsed, evaluation))
                p.start()
                if len(other_hp_combs) > 0:
                    for idx, other_hp_comb in enumerate(other_hp_combs):
                        hp_combination = construct(build_hp_comb)
                        hp_combination.update(other_hp_comb)
                        res = queue.get()
                        # print('\nidx: %s\nres: %s' % (idx, res))
                        # print('hp_combination:', hp_combination)
                        # print('res:', res)
                        self._handler.process_results(hp_combination, res, regime='several_launches')
                else:
                    hp_combination = construct(build_hp_comb)
                    res = queue.get()
                    self._handler.process_results(hp_combination, res, regime='several_launches')
                p.join()
        else:
            parsed = configure_args_for_launches(self, args_for_launches, list())
            queue = mp.Queue()
            # from some_useful_functions import nested2string
            # print('build_kwargs:', nested2string(build_kwargs))
            # print('parsed:', nested2string(parsed))
            self.mp_debug_flag += 1
            p = mp.Process(target=self._several_launches_without_rebuilding,
                           args=(queue, kwargs_for_building, session_specs, parsed, evaluation))
            p.start()
            if len(other_hp_combs) > 0:
                for idx, other_hp_comb in enumerate(other_hp_combs):
                    hp_combination = OrderedDict()
                    hp_combination.update(other_hp_comb)
                    res = queue.get()
                    # print('\nidx: %s\nres: %s' % (idx, res))
                    # print('hp_combination:', hp_combination)
                    # print('res:', res)
                    self._handler.process_results(hp_combination, res, regime='several_launches')
            else:
                pass
            p.join()

        self._handler.log_finish_time()
        self._handler.close()

    @staticmethod
    def _prepare_replica(replica, batch_generator_class, bpe_codes, batch_gen_args):
        if getattr(batch_generator_class, 'make_pairs', None) is not None:
            if bpe_codes is not None:
                with open(bpe_codes, 'r') as codes:
                    replica = prepare_for_bpe(replica)
                    bpe = BPE(codes)
                    replica = bpe.segment(replica)
                    replica = bpe_post_processing(replica)
                    replica = batch_generator_class.make_pairs(replica, batch_gen_args)
                    codes.close()
            else:
                replica = batch_generator_class.make_pairs(replica, batch_gen_args)
        else:
            replica = list(replica)
        return replica

    @staticmethod
    def _build_replica(replica):
        if isinstance(replica, str):
            return replica
        if isinstance(replica, list):
            if len(replica) == 0:
                return ''
            else:
                if isinstance(replica[0], str):
                    return ''.join(replica)
                if isinstance(replica[0], tuple):
                    return ''.join([''.join(p) for p in replica])

    def inference(self,
                  restore_path,
                  log_path,
                  vocabulary,
                  character_positions_in_vocabulary,
                  batch_generator_class,
                  additions_to_feed_dict=None,
                  gpu_memory=None,
                  allow_growth=False,
                  allow_soft_placement=False,
                  log_device_placement=False,
                  visible_device_list='',
                  appending=True,
                  temperature=0.,
                  first_speaker='human',
                  bpe_codes=None,
                  batch_gen_args=None):
        if additions_to_feed_dict is None:
            feed_dict_base = dict()
        else:
            feed_dict_base = dict()
            for addition in additions_to_feed_dict:
                feed_dict_base[self._hooks[addition['placeholder']]] = addition['value']

        create_path(log_path, file_name_is_in_path=True)
        if not appending:
            log_path = add_index_to_filename_if_needed(log_path)
        self._start_session(allow_soft_placement,
                            log_device_placement,
                            gpu_memory,
                            allow_growth,
                            visible_device_list)
        if restore_path is None:
            print_and_log('Skipping variables restoring. Continuing on current variables values', fn=log_path)
        else:
            self._session.run(tf.global_variables_initializer())
            self._restore_pupil(restore_path)
        self._hooks['reset_validation_state'].run(session=self._session)
        if first_speaker == 'human':
            human_replica = input('Human: ')
        else:
            human_replica = ''

        human_replica = self._prepare_replica(human_replica, batch_generator_class, bpe_codes, batch_gen_args)

        sample_prediction = self._hooks['validation_predictions']
        sample_input = self._hooks['validation_inputs']
        while not self._build_replica(human_replica) == 'FINISH':
            # print('(Environment.inference)human_replica:', human_replica)
            # print('(Environment.inference)self._build_replica(human_replica):', self._build_replica(human_replica))
            if len(human_replica) > 0:
                # print('(Environment.inference)human_replica:', human_replica)
                print_and_log('Human: ' + self._build_replica(human_replica), _print=False, fn=log_path)
                for char in human_replica:
                    feed = batch_generator_class.char2vec(char, character_positions_in_vocabulary, 0, 2)
                    feed_char = batch_generator_class.vec2char_fast(np.reshape(feed, (1, -1)), vocabulary)[0]
                    print('feed.shape:', feed.shape)
                    feed_dict = dict(feed_dict_base.items())
                    feed_dict[sample_input] = feed
                    excess_pred = sample_prediction.eval(feed_dict=feed_dict, session=self._session)
                    excess_char = batch_generator_class.vec2char(np.reshape(excess_pred, (1, -1)), vocabulary)[0]
                    print('char:%s|||feed_char:%s|||excess_char:%s|||' % (char, feed_char, excess_char))
            feed = batch_generator_class.char2vec('\n', character_positions_in_vocabulary, 0, 2)
            feed_dict = dict(feed_dict_base.items())
            feed_dict[sample_input] = feed
            prediction = sample_prediction.eval(feed_dict=feed_dict, session=self._session)
            if temperature != 0.:
                prediction = apply_temperature(prediction, -1, temperature)
                prediction = sample(prediction, -1)
            counter = 0
            char = batch_generator_class.vec2char(np.reshape(prediction, (1, -1)), vocabulary)[0]
            # print('char:', char)
            bot_replica = ''
            if char != '\n':
                bot_replica += char
            # print('ord(\'\\n\'):', ord('\n'))
            while char != '\n' and counter < 500:
                # print('char:', repr(char))
                # print('prediction:\n', prediction)
                feed = batch_generator_class.pred2vec(prediction, 1, 2, batch_gen_args)
                # print('feed:\n', feed)
                # print('prediction after sampling:', prediction)
                # print('feed:', feed)
                feed_dict = dict(feed_dict_base.items())
                feed_dict[sample_input] = feed
                prediction = sample_prediction.eval(feed_dict=feed_dict, session=self._session)
                # print('prediction before sampling:', prediction)
                if temperature != 0.:
                    prediction = apply_temperature(prediction, -1, temperature)
                    # print('prediction after temperature:', prediction)
                    prediction = sample(prediction, -1)
                char = batch_generator_class.vec2char(np.reshape(prediction, (1, -1)), vocabulary)[0]
                if char != '\n':
                    # print('char != \'\\n\', counter = %s' % counter)
                    # print('ord(char):', ord(char))
                    bot_replica += char
                counter += 1
            print_and_log('Bot: ' + self._build_replica(bot_replica), fn=log_path)
            feed = batch_generator_class.char2vec('\n', character_positions_in_vocabulary, 1, 2)
            feed_dict = dict(feed_dict_base.items())
            feed_dict[sample_input] = feed
            _ = sample_prediction.eval(feed_dict=feed_dict, session=self._session)

            human_replica = input('Human: ')
            human_replica = self._prepare_replica(human_replica, batch_generator_class, bpe_codes, batch_gen_args)
        with open(log_path, 'a') as fd:
            fd.write('\n*********************')
        self._close_session()

    def _feed_replica(self, replica, batch_generator_class,
                      character_positions_in_vocabulary, temperature,
                      feed_dict_base, speaker, bpe_codes, batch_gen_args):
        replica = self._prepare_replica(replica, batch_generator_class, bpe_codes, batch_gen_args)
        if speaker == 'bot':
            flag = 1
        else:
            flag = 0
        sample_input = self._hooks['validation_inputs']
        sample_prediction = self._hooks['validation_predictions']
        for char in replica:
            feed = batch_generator_class.char2vec(char, character_positions_in_vocabulary, flag, 2)
            # print('feed.shape:', feed.shape)
            feed_dict = dict(feed_dict_base.items())
            feed_dict[sample_input] = feed
            _ = sample_prediction.eval(feed_dict=feed_dict, session=self._session)
        feed = batch_generator_class.char2vec('\n', character_positions_in_vocabulary, flag, 2)
        feed_dict = dict(feed_dict_base.items())
        feed_dict[sample_input] = feed
        prediction = sample_prediction.eval(feed_dict=feed_dict, session=self._session)
        if temperature != 0.:
            prediction = apply_temperature(prediction, -1, temperature)
            prediction = sample(prediction, -1)
        return prediction

    def _generate_replica(self, prediction, batch_generator_class, vocabulary,
                          character_positions_in_vocabulary, temperature, feed_dict_base, speaker, batch_gen_args):
        if speaker == 'bot':
            flag = 1
        else:
            flag = 0
        counter = 0
        char = None
        bot_replica = ""
        sample_input = self._hooks['validation_inputs']
        sample_prediction = self._hooks['validation_predictions']
        # print('ord(\'\\n\'):', ord('\n'))
        while char != '\n' and counter < 250:
            feed = batch_generator_class.pred2vec(prediction, flag, 2, batch_gen_args)
            # print('prediction after sampling:', prediction)
            # print('feed:', feed)
            feed_dict = dict(feed_dict_base.items())
            feed_dict[sample_input] = feed
            prediction = sample_prediction.eval(feed_dict=feed_dict, session=self._session)
            # print('prediction before sampling:', prediction)
            if temperature != 0.:
                prediction = apply_temperature(prediction, -1, temperature)
                # print('prediction after temperature:', prediction)
                prediction = sample(prediction, -1)
            char = batch_generator_class.vec2char(np.reshape(feed, (1, -1)), vocabulary)[0]
            if char != '\n':
                # print('char != \'\\n\', counter = %s' % counter)
                # print('ord(char):', ord(char))
                bot_replica += char
            counter += 1
        feed = batch_generator_class.char2vec('\n', character_positions_in_vocabulary, flag, 2)
        feed_dict = dict(feed_dict_base.items())
        feed_dict[sample_input] = feed
        _ = sample_prediction.eval(feed_dict=feed_dict, session=self._session)
        return bot_replica, prediction

    def _one_chat(
            self,
            kwargs_for_building,
            restore_path,
            # log_path,
            vocabulary,
            character_positions_in_vocabulary,
            batch_generator_class,
            additions_to_feed_dict,
            gpu_memory,
            allow_growth,
            temperature,
            bpe_codes,
            batch_gen_args,
            inq,
            outq):
        # print('entered _one_chat')
        self._build_pupil(kwargs_for_building)
        if additions_to_feed_dict is None:
            feed_dict_base = dict()
        else:
            feed_dict_base = dict()
            for addition in additions_to_feed_dict:
                feed_dict_base[self._hooks[addition['placeholder']]] = addition['value']
        self._start_session(False,
                            False,
                            gpu_memory,
                            allow_growth,
                            '')
        self._session.run(tf.global_variables_initializer())
        self._restore_pupil(restore_path)
        self._hooks['reset_validation_state'].run(session=self._session)
        greeting = 'Здравствуйте, я бот.'
        # print_and_log('Bot: ' + greeting, _print=False, fn=log_path)
        print('(Environment.one_chat)inq:', inq)
        _ = inq.get()
        # _ = inq.get(block=False)
        outq.put(greeting)
        # print('greeting is put in queue')
        _ = self._feed_replica(
            greeting, batch_generator_class,
            character_positions_in_vocabulary, temperature, feed_dict_base, 'bot', bpe_codes, batch_gen_args)
        timeshot = time.time()
        try:
            human_replica = inq.get(timeout=300)
        except queue.Empty:
            human_replica = ''
            pass

        while not human_replica == '/end' and time.time() - timeshot < 290:
            # print('(start while)time.time() - timeshot =', time.time() - timeshot)
            # print('(start while)time.time() =', time.time())
            # print('(start while)timeshot =', timeshot)
            if human_replica != '':
                # print_and_log('Human: ' + human_replica, _print=False, fn=log_path)
                prediction = self._feed_replica(
                    human_replica, batch_generator_class,
                    character_positions_in_vocabulary, temperature,
                    feed_dict_base, 'human', bpe_codes, batch_gen_args
                )
                bot_replica, prediction = self._generate_replica(
                    prediction, batch_generator_class, vocabulary,
                    character_positions_in_vocabulary, temperature, feed_dict_base, 'bot', batch_gen_args)
                # print_and_log('Bot: ' + bot_replica, _print=False, fn=log_path)
                outq.put(bot_replica)
                timeshot = time.time()
            try:
                human_replica = inq.get(timeout=300)
            except queue.Empty:
                human_replica = ''
            # print('(end while)time.time() - timeshot =', time.time() - timeshot)
            # print('(end while)time.time() =', time.time())
            # print('(end while)timeshot =', timeshot)
        # print('reached -1')
        outq.put(-1)

    def telegram(self,
                 kwargs_for_building,
                 restore_path,
                 log_path,
                 vocabulary,
                 character_positions_in_vocabulary,
                 batch_generator_class,
                 additions_to_feed_dict=None,
                 gpu_memory=None,
                 allow_growth=True,
                 temperature=0.,
                 bpe_codes=None,
                 batch_gen_args=None):
        if len(log_path) > 4 and log_path[-4:] == '.txt':
            create_path(log_path, file_name_is_in_path=True)
        else:
            create_path(log_path, file_name_is_in_path=False)

        inqs = dict()
        outqs = dict()
        ps = dict()
        file_names = dict()

        writer = csv.writer(sys.stdout, quoting=csv.QUOTE_NONNUMERIC)
        read_list = [sys.stdin]
        try:
            while read_list:
                # print('entered while loop')
                ready = select.select(read_list, [], [], 0)[0]
                if ready:
                    # print('ready:', ready)
                    text = ready[0].readline()
                    # print('text:', text)
                    row = csv.reader([text]).__next__()
                    chat_id_has_corr_format = is_int(row[0])
                    if chat_id_has_corr_format:
                        chat_id, question = int(row[0]), row[1]
                        if chat_id not in inqs:
                            # print('chat_id not in inqs')
                            if len(log_path) > 4 and log_path[-4:] == '.txt':
                                file_name = add_index_to_filename_if_needed(log_path, index=0)
                            else:
                                file_name = add_index_to_filename_if_needed(log_path + '/chat.txt', index=0)
                            file_names[chat_id] = file_name
                            inqs[chat_id] = mp.Queue()
                            # print('(Environment.telegram)inqs[chat_id]:', inqs[chat_id])
                            outqs[chat_id] = mp.Queue()
                            ps[chat_id] = mp.Process(target=self._one_chat,
                                                     args=(kwargs_for_building, restore_path, vocabulary,
                                                           character_positions_in_vocabulary,
                                                           batch_generator_class, additions_to_feed_dict, gpu_memory,
                                                           allow_growth, temperature, bpe_codes, batch_gen_args,
                                                           inqs[chat_id], outqs[chat_id]))
                            # print('(Environment.telegram)question:', question)
                            inqs[chat_id].put(question)
                            # print('(Environment.telegram)inqs[chat_id]:', inqs[chat_id])
                            ps[chat_id].start()
                            # print('ps:', ps)
                        else:
                            inqs[chat_id].put(question)


                        if question != '/start' and question != '/end':
                            print_and_log('Human: ' + question, _print=False, fn=file_names[chat_id])
                # print('reached outqs loop')
                for chat_id in list(outqs.keys()):
                    try:
                        bot_replica = outqs[chat_id].get(block=False)
                    except queue.Empty:
                        bot_replica = -2
                    if bot_replica == -1:
                        # print(-1)
                        ps[chat_id].join()
                        if ps[chat_id].is_alive():
                            print('WARNING! Could not join process for chat %s' % chat_id)
                            ps[chat_id].terminate()
                            ps[chat_id].join()
                        del ps[chat_id]
                        del inqs[chat_id]
                        del outqs[chat_id]
                        del file_names[chat_id]
                    elif bot_replica != -2:
                        print_and_log('Bot: ' + bot_replica, _print=False, fn=file_names[chat_id])
                        writer.writerow([chat_id, bot_replica, "", "/start", "Ты дурак.", "/end"])
                        sys.stdout.flush()

        except KeyboardInterrupt:
            for inq in inqs.values():
                inq.put('/end')
            for chat_id, outq in outqs.items():
                try:
                    flag = outq.get(timeout=.01)
                    # print('1 try')
                    while flag != -1:
                        flag = outq.get(timeout=.01)
                        # print('another try')
                except queue.Empty:
                    print('WARNING! Process termination flag was not received for chat %s' % chat_id)
                ps[chat_id].join(timeout=.01)
                if ps[chat_id].is_alive():
                    print('WARNING! Could not join process for chat %s' % chat_id)
                    ps[chat_id].terminate()
                    ps[chat_id].join()

    def generate_discriminator_dataset(self,
                                       num_examples,
                                       num_repeats,
                                       dataset_text,
                                       gen_max_length,
                                       fuse_stop,
                                       restore_path,
                                       save_path,
                                       vocabulary=None,
                                       additions_to_feed_dict=None,
                                       gpu_memory=None):
        if additions_to_feed_dict is None:
            additions_to_feed_dict = dict()
        if vocabulary is None and self._vocabulary is None:
            raise InvalidArgumentError(
                'Vocabulary has to be provided either to Environment constructor' +
                ' or to generate_discriminator_dataset method', None, 'vocabulary', 'list of chars')
        elif vocabulary is None:
            vocabulary = self._vocabulary

        all_phrases = dataset_text.split('\n')[1:-1]
        # print('dataset_text:', dataset_text)
        # print('all_phrases:', all_phrases)
        replicas = all_phrases[:-1]
        answers = all_phrases[1:]
        num_replicas = len(replicas)
        # print('num_replicas:', num_replicas)
        interval = num_replicas // num_examples
        # print('interval:', interval)
        used_replicas = list()
        used_answers = list()
        # for replica in replicas:
        #     print('ord(replica[-1])', ord(replica[-1]))
        if interval == 0:
            for replica, answer in zip(replicas, answers):
                used_replicas.append(replica[1:])
                used_answers.append(answer[1:])
        else:
            for i in range(num_examples):
                # print('ord(replicas[-1])', ord(replicas[-1]))
                used_replicas.append(replicas[i*interval][1:])
                used_answers.append(answers[i*interval][1:])
        # print('used_replicas:', used_replicas)
        # print('used_answers:', used_answers)
        create_path(save_path, False)
        fuses_fd = open(add_index_to_filename_if_needed(save_path + '/fuses.txt'), 'w', encoding='utf-8')
        correct_answers_fd = open(add_index_to_filename_if_needed(save_path + '/correct.txt'), 'w', encoding='utf-8')
        fuses = list()
        for replica, answer in zip(used_replicas, used_answers):
            fuses.append({'text': replica + '\n', 'num_repeats': num_repeats,
                          'max_num_of_chars': gen_max_length, 'fuse_stop': fuse_stop})
            fuses_fd.write(replica + '\n')
            correct_answers_fd.write(answer + '\n')
        fuses_fd.close()
        correct_answers_fd.close()

        fuse_results, _ = self.test(
            restore_path=restore_path,
            print_results=False,
            vocabulary=vocabulary,
            additions_to_feed_dict=additions_to_feed_dict,
            printed_result_types=None,
            fuses=fuses,
            random=None,
            gpu_memory=gpu_memory)

        generated_fd = open(add_index_to_filename_if_needed(save_path + '/generated.txt'), 'w')
        generated_text = ''
        for fuse_res in fuse_results:
            for phrase_idx, phrase in enumerate(fuse_res['results']):
                phrase = re.sub("[\t\n]+", '', phrase)
                generated_fd.write(phrase[1:])
                generated_text += phrase[1:]
                # print('phrase_idx:', phrase_idx, 'length:', len(phrase))
                if phrase_idx < len(fuse_res['results']) - 1:
                    # print('phrase_idx:', phrase_idx)
                    generated_text += '\t'
                    num_chars = generated_fd.write("\t")
                    # print('num_chars:', num_chars)
            generated_fd.write('\n')
            generated_text += '\n'
        generated_fd.close()
        # fd = open(save_path + '/generated.txt', 'r', encoding='utf-8')
        # file_text = fd.read()
        # print('file_text:', file_text)

        # print('generated_text:', generated_text)

    def _store_launch_parameters(self, model_type, **kwargs):
        if model_type == 'pupil':
            self.current_pupil_launch_parameters = kwargs
        elif model_type == 'optimizer':
            self.current_optimizer_launch_parameters = kwargs



