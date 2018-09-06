import csv
import multiprocessing as mp
import queue
import random
import re
import select
import sys
import os
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from learning_to_learn.args_parsing import parse_1_set_of_kwargs, parse_train_method_arguments, \
    formalize_and_create_insertions_for_build_hps, formalize_and_create_insertions_for_other_hps, \
    create_all_args_for_launches, configure_args_for_launches, parse_train_optimizer_method_arguments
from learning_to_learn.bpe import prepare_for_bpe, bpe_post_processing
from tensorflow.python import debug as tf_debug
from learning_to_learn.tensors import identity_tensor
from learning_to_learn.useful_functions import InvalidArgumentError
from learning_to_learn.useful_functions import construct, add_index_to_filename_if_needed, match_two_dicts, \
    create_path, check_if_key_in_nested_dict, add_missing_to_list, print_and_log, apply_temperature, sample, is_int, \
    create_distribute_map, nth_element_of_sequence_of_sequences, get_elem_from_nested, form_combinations_from_dicts

from learning_to_learn.handler import Handler
from subword_nmt.apply_bpe import BPE


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
        # print("(Controller.__init__)specifications:", specifications)
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
        elif self._specifications['type'] == 'adaptive_change':
            self.get = self._adaptive_change
            self._specifications = dict(self._specifications)
            self._specifications['num_points_since_best_res'] = 0
            self._specifications['value'] = self._specifications['init']
            self._specifications['last_fire_line_length'] = -1
            self._init_ops_for_adaptive_controller()
        elif self._specifications['type'] == 'fire_at_best':
            self.get = self._fire_at_best
            self._init_ops_for_adaptive_controller()
        elif self._specifications['type'] == 'while_progress':
            self.get = self._while_progress
            self._specifications = dict(self._specifications)
            self._specifications['num_points_since_best_res'] = 0
            self._specifications['cur_made_prog'] = False
            self._specifications['prev_made_prog'] = True
            self._init_ops_for_adaptive_controller()
            self._specifications['current_value'] = self._specifications['changing_parameter_controller'].get()
        elif self._specifications['type'] == 'while_progress_no_changing_parameter':
            self.get = self._while_progress_no_changing_parameter
            self._specifications = dict(self._specifications)
            self._specifications['num_points_since_best_res'] = 0
            self._init_ops_for_adaptive_controller()
            
    def _init_ops_for_adaptive_controller(self):
        if 'direction' not in self._specifications:
            self._specifications['direction'] = 'down'
        if self._specifications['direction'] == 'down':
            self._specifications['comp_func'] = self._comp_func_gen(min)
        else:
            self._specifications['comp_func'] = self._comp_func_gen(max)
        # print(self._storage)
        self._specifications['line'] = get_elem_from_nested(
            self._storage,
            self._specifications['path_to_target_metric_storage']
        )
        self._specifications['last_fire_line_length'] = -1

    @staticmethod
    def _comp_func_gen(comp):
        def f(line):
            if len(line) == 0:
                return True
            elif len(line) == 1:
                return comp(line) == line[-1]
            else:
                return comp(line) == line[-1] and comp(line[:-1]) != line[-1]
        return f

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

    def _adaptive_change(self):
        specs = self._specifications
        if specs['comp_func'](specs['line']):
            specs['num_points_since_best_res'] = 0
            return specs['value']
        else:
            if specs['last_fire_line_length'] < len(specs['line']):
                specs['num_points_since_best_res'] += 1
                specs['last_fire_line_length'] = len(specs['line'])
                if specs['num_points_since_best_res'] > specs['max_no_progress_points']:
                    specs['value'] *= specs['decay']
                    specs['num_points_since_best_res'] = 0
            return specs['value']

    def _fire_at_best(self):
        specs = self._specifications
        # print("(Controller._fire_at_best)specs['comp_func'](specs['line']):", specs['comp_func'](specs['line']))
        # print("(Controller._fire_at_best)specs['line'][-1]:", specs['line'][-1])
        # print("(Controller._fire_at_best)specs['last_fire_line_length']:", specs['last_fire_line_length'])
        # print("(Controller._fire_at_best)len(specs['line']):", len(specs['line']))
        if specs['comp_func'](specs['line']) and specs['last_fire_line_length'] < len(specs['line']):
            specs['last_fire_line_length'] = len(specs['line'])
            return True
        else:
            return False

    def _while_progress(self):  # for example if learning does not bring improvement return False
        specs = self._specifications
        value = specs['changing_parameter_controller'].get()
        if specs['current_value'] == value:
            if specs['comp_func'](specs['line']):
                specs['num_points_since_best_res'] = 0
                specs['cur_made_prog'] = True
                ret = True
            else:
                if not specs['prev_made_prog'] and \
                                specs['num_points_since_best_res'] > \
                                specs['max_no_progress_points']:
                    ret = False
                else:
                    if specs['last_fire_line_length'] < len(specs['line']):
                        specs['num_points_since_best_res'] += 1
                    ret = True
        else:
            if not specs['cur_made_prog'] and not specs['prev_made_prog']:
                return False
            else:
                specs['prev_made_prog'] = specs['cur_made_prog']
                specs['cur_made_prog'] = False
                specs['num_points_since_best_res'] = 0
                specs['current_value'] = value
                ret = True
        specs['last_fire_line_length'] = len(specs['line'])
        return ret

    def _while_progress_no_changing_parameter(self):
        specs = self._specifications
        # print("(Controller._while_progress_no_changing_parameter)specs['num_points_since_best_res']:",
        #       specs['num_points_since_best_res'])
        if specs['comp_func'](specs['line']):
            specs['num_points_since_best_res'] = 0
            ret = True
        else:
            if specs['num_points_since_best_res'] > specs['max_no_progress_points']:
                ret = False
            else:
                if specs['last_fire_line_length'] < len(specs['line']):
                    specs['num_points_since_best_res'] += 1
                ret = True
        specs['last_fire_line_length'] = len(specs['line'])
        return ret


    @staticmethod
    def _always_false():
        return False

    @property
    def name(self):
        return self._specifications['name']


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

        tensor_schedule = dict(
            train_print_tensors=dict(),
            train_save_tensors=dict(),
            train_print_text_tensors=dict(),
            train_save_text_tensors=dict(),
            train_summary_tensors=dict()
        )

        valid_tensor_schedule = {'valid_print_tensors': dict(),
                                 'valid_save_text_tensors': dict()}

        fuse_tensors = {'fuse_print_tensors': dict(), 'fuse_save_tensors': dict()}
        example_tensors = {'example_print_tensors': dict(), 'example_save_tensors': dict()}

        # Every results_collect_interval-th step BPC, accuracy, perplexity are collected
        # Every print_per_collected-th point containing BPC, accuracy and perplexity is printed
        # Together with every example_per_print-th point example is printed
        default_collected_while_training = {
            'results_collect_interval': 100,
            'print_per_collected': 1,
            'example_per_print': 1
        }

        optimizer_inference_default_collected_while_training = {
            'opt_inf_results_collect_interval': 10,
            'opt_inf_print_per_collected': 1,
            'opt_inf_example_per_print': 1
        }

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
            session_specs=dict(
                allow_soft_placement=False,
                gpu_memory=None,
                allow_growth=False,
                log_device_placement=False,
                visible_device_list=""
            ),
            start_specs=dict(
                restore_path=None,
                with_meta_optimizer=False,
                restore_optimizer_path=None,
                save_path=None,
                result_types=self.put_result_types_in_correct_order(
                    ['loss', 'perplexity', 'accuracy']),
                summary=False,
                add_graph_to_summary=False,
                batch_generator_class=self._default_batch_generator,
                vocabulary=self._vocabulary
            ),
            run=dict(
                train_specs=dict(
                    learning_rate=construct(default_learning_rate_control),
                    additions_to_feed_dict=list(),
                    stop={'type': 'limit_steps', 'limit': 10000, 'name': 'stop'},
                    train_dataset=default_dataset,  # list of 2 elements. First is text string, the second is name
                    batch_size={'type': 'fixed', 'value': 64, 'name': 'batch_size'},
                    train_batch_kwargs=dict(),
                    checkpoint_steps=None,
                    debug=None,
                    validation_datasets=None,
                    validation_additions_to_feed_dict=list(),
                    validation_batch_size=1,
                    valid_batch_kwargs=dict(),
                    validate_tokens_by_chars=False,
                    no_validation=False
                ),
                schedule=dict(
                    to_be_collected_while_training=construct(default_collected_while_training),
                    printed_result_types=self.put_result_types_in_correct_order(
                        ['loss']),
                    printed_controllers=['learning_rate'],
                    fuses=None,
                    fuse_tensors=construct(fuse_tensors),
                    example_length=None,
                    example_tensors=construct(example_tensors),
                    replicas=None,
                    random={'number_of_runs': 5, 'length': 80},
                    train_tensor_schedule=construct(tensor_schedule),
                    validation_tensor_schedule=construct(valid_tensor_schedule)
                )
            )
        )

        self._default_train_optimizer_method_args = dict(
            session_specs=dict(
                allow_soft_placement=False,
                gpu_memory=None,
                allow_growth=False,
                log_device_placement=False,
                visible_device_list=""
            ),
            start_specs=dict(
                restore_optimizer_path=None,
                save_path=None,
                result_types=self.put_result_types_in_correct_order(
                    ['loss']),
                summary=False,
                add_graph_to_summary=False,
                batch_generator_class=self._default_batch_generator,
                vocabulary=self._vocabulary
            ),
            run=dict(
                train_specs=dict(
                    learning_rate=construct(default_learning_rate_control),
                    additions_to_feed_dict=list(),
                    stop={'type': 'limit_steps', 'limit': 10000, 'name': 'stop'},

                    pupil_restore_paths=[None],
                    num_exercises=10,
                    reset_period=None,
                    batch_gen_init_is_random=True,
                    one_batch_gen=False,
                    share_train_data=False,

                    restore_paths_datasets_map=None,

                    train_datasets=[default_dataset],
                    batch_size={'type': 'fixed', 'value': 32, 'name': 'batch_size'},
                    train_batch_kwargs=dict(),

                    checkpoint_steps=None,
                    debug=None
                ),
                optimizer_inference=dict(
                    opt_inf_is_performed=False,
                    opt_inf_stop=None,
                    opt_inf_pupil_restore_paths=None,
                    opt_inf_additions_to_feed_dict=None,
                    opt_inf_to_be_collected_while_training=construct(
                        optimizer_inference_default_collected_while_training),
                    opt_inf_train_datasets=[default_dataset],
                    opt_inf_validation_datasets=None,
                    validation_additions_to_feed_dict=list(),
                    validation_batch_size=1,
                    valid_batch_kwargs=dict(),
                    validate_tokens_by_chars=False,
                    no_validation=False,
                    fuses=None,
                    fuse_tensors=construct(fuse_tensors),
                    example_length=None,
                    example_tensors=construct(example_tensors),
                    replicas=None,
                    random={'number_of_runs': 5, 'length': 80},
                    opt_inf_train_tensor_schedule=construct(tensor_schedule),
                    opt_inf_validation_tensor_schedule=construct(valid_tensor_schedule)
                ),
                schedule=dict(
                    to_be_collected_while_training=construct(default_collected_while_training),
                    printed_result_types=self.put_result_types_in_correct_order(
                        ['loss']),
                    printed_controllers=['learning_rate'],
                    train_tensor_schedule=construct(tensor_schedule),
                )
            )
        )

        self._default_test_method_args = dict(
            session_specs=dict(
                allow_soft_placement=False,
                gpu_memory=None,
                allow_growth=False,
                log_device_placement=False,
                visible_device_list=""
            ),
            start_specs=dict(
                restore_path=None,
                save_path=None,
                print_results=True,
                result_types=self.put_result_types_in_correct_order(
                    ['loss', 'perplexity', 'accuracy']
                ),
                verbose=True,
                batch_generator_class=self._default_batch_generator,
                vocabulary=self._vocabulary
            ),
            work=dict(
                additions_to_feed_dict=list(),
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
        self._current_place_for_result_saving = self._storage
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
        # print('(Environment._build_pupil)default_hooks:', default_hooks)
        self._hooks.update(default_hooks)
        # self._register_default_builders()

    def build_optimizer(self, **kwargs):
        self._meta_optimizer_class.check_kwargs(**kwargs)
        self.current_pupil_build_parameters = kwargs
        self._meta_optimizer = self._meta_optimizer_class(self._pupil, **kwargs)
        default_hooks = self._meta_optimizer.get_default_hooks()
        # print("(Environment.build_optimizer)default_hooks:", default_hooks)
        self._hooks.update(default_hooks)

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
    def default_train_optimizer_method_args(self):
        return construct(self._default_train_optimizer_method_args)

    @default_train_optimizer_method_args.setter
    def default_train_optimizer_method_args(self, update):
        """update is a dictionary which should match keys of self._pupil_default_training"""
        self._update_dict(self._default_train_optimizer_method_args, update)

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
        if method_name == 'train_optimizer':
            return self.default_train_optimizer_method_args
        return None

    def _start_session(self, allow_soft_placement, log_device_placement, gpu_memory, allow_growth, visible_device_list):
        """Starts new session with specified parameters. If there is opend session closes it"""
        if self._session is not None:
            print('Warning: there is an opened session already. Closing it')
            self._session.close()
        # print('(_start_session)gpu_memory:', gpu_memory)
        # print('(_start_session)allow_growth:', allow_growth)
        config = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=gpu_memory,
                allow_growth=allow_growth,
                visible_device_list=visible_device_list
            ),
            log_device_placement=log_device_placement
        )
        # config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
        self._session = tf.Session(config=config)

    def _close_session(self):
        self._session.close()
        self._session = None

    def init_storage(self, dataset_name, **kwargs):
        self._current_place_for_result_saving[dataset_name] = dict()
        d = self._current_place_for_result_saving[dataset_name]
        for key, value in kwargs.items():
            d[key] = value

    @staticmethod
    def create_train_pupil_storage(stor, result_types, dataset_names):
        for dataset_name in dataset_names:
            stor[dataset_name] = dict()
        for res_type in result_types + ['steps']:
            for dataset_name in dataset_names:
                stor[dataset_name][res_type] = list()
        return stor

    def init_meta_optimizer_training_storage(
            self, opt_inf_pupil_names=None,
            train_init=None,
            # opt_inf_init=None,
            wipe_storage=False
    ):
        if self._current_place_for_result_saving is None or wipe_storage:
            self._current_place_for_result_saving = dict()
        self._current_place_for_result_saving['step'] = 0
        if train_init is not None:
            self._current_place_for_result_saving['train'] = construct(train_init)
        # example of optimizer inference results during training
        # self._storage[<pupil_name>][<regime>(train or validation)][<result type>][<optimizer step index>]
        # [<pupil step index>]

        # opt_inf_init is not None and
        if opt_inf_pupil_names is not None:
            for name in opt_inf_pupil_names:
                self._current_place_for_result_saving[name] = dict(
                    steps=list(),
                    results=list()   # list of train storages with results
                )

    def append_to_storage(self, dataset_name, **kwargs):
        # print("(Environment.append_to_storage)self._storage:", self._storage)
        # print(
        #     "(Environment.append_to_storage)self._current_place_for_result_saving:",
        #     self._current_place_for_result_saving
        # )
        if dataset_name is not None:
            d = self._current_place_for_result_saving[dataset_name]
        else:
            d = self._current_place_for_result_saving
        for key, value in kwargs.items():
            d[key].append(value)

    # def append_to_optimizer_inference_storage(self, pupil_name, regime, optimizer_training_step, **kwargs):
    #     if optimizer_training_step in self._current_place_for_result_saving[pupil_name]['steps']:
    #         index = self._current_place_for_result_saving[pupil_name]['steps'].index(optimizer_training_step)
    #     else:
    #         index = len(self._current_place_for_result_saving[pupil_name]['steps'])
    #         for key, value in kwargs.items():
    #             self._current_place_for_result_saving[pupil_name][regime][key].append(list())
    #     for key, value in kwargs.items():
    #         self._current_place_for_result_saving[pupil_name][regime][key][index].append(value)

    def flush_storage(self):
        self._current_place_for_result_saving = {'step': None}

    def set_in_storage(self, **kwargs):
        for key, value in kwargs.items():
            self._current_place_for_result_saving[key] = value

    def check_if_key_in_storage(self, keys):
        return check_if_key_in_nested_dict(self._current_place_for_result_saving, keys)

    def dataset_in_storage(self, dataset_name):
        return dataset_name in self._current_place_for_result_saving

    def _create_checkpoint(self, step, checkpoints_path, model_type='pupil'):
        path = checkpoints_path + '/' + str(step)
        print('\nCreating %s checkpoint at %s' % (model_type, path))
        if model_type == 'pupil':
            self._hooks['saver'].save(self._session, path)
        elif model_type == 'optimizer':
            # print("(Environment._create_checkpoint)self._hooks['meta_optimizer_saver']:",
            #       self._hooks['meta_optimizer_saver'])
            self._hooks['meta_optimizer_saver'].save(self._session, path)

    def _restore_pupil(self, restore_path, verbose=True):
        # print("(Environment._restore_pupil)self._hooks:", self._hooks)
        if restore_path is not None:
            if verbose:
                print('restoring pupil from %s' % restore_path)
            self._hooks['saver'].restore(self._session, restore_path)

    def _restore_meta_optimizer(self, restore_path):
        if restore_path is not None:
            print('restoring meta optimizer from %s' % restore_path)
            # print("(Environment._restore_meta_optimizer)self._hooks['meta_optimizer_saver']:",
            #       self._hooks['meta_optimizer_saver'])
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
        # print('all_tensor_aliases:', all_tensor_aliases)
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
        empty_batch_gen = batch_generator_class('', 1, **work['valid_batch_kwargs'])
        if work['fuses'] is not None:
            fuse_res = self._on_fuses(empty_batch_gen,
                                      work['fuses'],
                                      additional_feed_dict=add_feed_dict)
        else:
            fuse_res = None

        validation_datasets = work['validation_datasets']
        # print("(Environment.test)work['valid_batch_kwargs']:", work['valid_batch_kwargs'])
        print("Testing!")
        for validation_dataset in validation_datasets:
            print("Validation dataset name:", validation_dataset[1])
            print("Validation dataset_size:")
            if len(validation_dataset[0]) > 100:
                print(len(validation_dataset[0]))
            else:
                print('Either dataset is not text or dataset size is less than 100:', len(validation_dataset[0]))
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
                    vec = batch_generator.char2vec(char, batch_generator.character_positions_in_vocabulary, None, None)
                    feed_dict = {
                        self._hooks['validation_inputs']: np.reshape(
                            vec, self._hooks['validation_inputs'].get_shape().as_list())
                    }
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
                        vec = batch_generator.pred2vec(fuse_res[0], None, None, None)
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
                        vec = batch_generator.pred2vec(fuse_res[0], None, None, None)
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
        for c_idx in range(min(example_length, example_batches.get_num_batches()) + 1):
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

    def _validate(
            self,
            batch_generator_class,
            validation_dataset,
            validation_batch_size,
            valid_batch_kwargs,
            training_step=None,
            additional_feed_dict=None,
            save_to_file=None,
            save_to_storage=None,
            print_results=None
    ):
        if additional_feed_dict is None:
            additional_feed_dict = dict()
        # print('valid_batch_kwargs:', valid_batch_kwargs)
        if 'reset_validation_state' in self._hooks:
            self._session.run(self._hooks['reset_validation_state'])
        # print('batch_generator_class:', batch_generator_class)
        valid_batches = batch_generator_class(validation_dataset[0], validation_batch_size, **valid_batch_kwargs)
        length = valid_batches.get_num_batches()
        inputs, labels = valid_batches.next()
        step = 0
        self._handler.start_accumulation(validation_dataset[1], training_step=training_step)
        # print("(Environment._validate/before loop)self._current_place_for_result_saving:",
        #       self._current_place_for_result_saving)
        while step < length:
            validation_operations = self._handler.get_tensors('validation', step)
            feed_dict = {self._hooks['validation_inputs']: inputs,
                         self._hooks['validation_labels']: labels}
            if isinstance(additional_feed_dict, dict):
                feed_dict.update(additional_feed_dict)
            # print("(Environment._validate)validation_operations:", validation_operations)
            # print("(Environment._validate)feed_dict:", feed_dict)
            valid_res = self._session.run(validation_operations, feed_dict=feed_dict)
            self._handler.process_results(training_step, valid_res, regime='validation')
            step += 1
            if step < length:
                inputs, labels = valid_batches.next()

        # print("(Environment._validate/after loop)self._current_place_for_result_saving:",
        #       self._current_place_for_result_saving)
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
        # print('batch_generator_class:', batch_generator_class)
        valid_batches = batch_generator_class(validation_dataset[0], validation_batch_size, **valid_batch_kwargs)
        length = valid_batches.get_num_batches()
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

    def _check_if_validation_is_needed(self, run_specs_set, where_no_validation_key):
        """This method is not finished yet. Fuses, random and replicas should also be taken in account"""
        validation_is_needed = False
        for run_specs in run_specs_set:
            validation_is_needed = validation_is_needed or \
                                   not run_specs[where_no_validation_key]['no_validation']
        return validation_is_needed

    def _all_tensor_aliases_from_train_method_arguments(self, args_for_launches, evaluation=None):
        start_specs_for_launches, run_specs_for_launches = zip(*args_for_launches)
        list_of_required_tensors_aliases = list()
        result_types_for_launches = list()
        for start_specs in start_specs_for_launches:
            result_types_for_launches = add_missing_to_list(
                result_types_for_launches, start_specs['result_types'])
        list_of_required_tensors_aliases.extend(result_types_for_launches)
        for start_specs, run_specs_set in zip(start_specs_for_launches, run_specs_for_launches):
            # if not start_specs['with_meta_optimizer']:
            if self._check_if_validation_is_needed(run_specs_set, 'train_specs'):
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

    def _all_tensor_aliases_from_train_meta_optimizer_method_arguments(self, args_for_launches):
        start_specs_for_launches, run_specs_for_launches = zip(*args_for_launches)
        list_of_required_tensors_aliases = list()
        result_types_for_launches = list()
        for start_specs in start_specs_for_launches:
            result_types_for_launches = add_missing_to_list(
                result_types_for_launches, start_specs['result_types'])
        list_of_required_tensors_aliases.extend(result_types_for_launches)
        for start_specs, run_specs_set in zip(start_specs_for_launches, run_specs_for_launches):
            # if not start_specs['with_meta_optimizer']:
            if self._check_if_validation_is_needed(run_specs_set, 'optimizer_inference'):
                for result_type in start_specs['result_types']:
                    list_of_required_tensors_aliases.append('validation_' + result_type)
        for run_specs_set in run_specs_for_launches:
            for run_specs in run_specs_set:
                train_aliases = self._get_all_tensors_from_schedule(
                    run_specs['schedule']['train_tensor_schedule'])
                list_of_required_tensors_aliases = add_missing_to_list(
                    list_of_required_tensors_aliases, train_aliases)
                valid_aliases = self._get_all_tensors_from_schedule(
                    run_specs['optimizer_inference']['opt_inf_train_tensor_schedule'])
                list_of_required_tensors_aliases = add_missing_to_list(
                    list_of_required_tensors_aliases, valid_aliases)
                valid_aliases = self._get_all_tensors_from_schedule(
                    run_specs['optimizer_inference']['opt_inf_validation_tensor_schedule'])
                list_of_required_tensors_aliases = add_missing_to_list(
                    list_of_required_tensors_aliases, valid_aliases)
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

    def _train(
            self,
            run_specs,
            checkpoints_path,
            batch_generator_class,
            with_meta_optimizer,
            init_step=0,
            # storage=None,
    ):
        """It is a method that does actual training and responsible for one training pass through dataset. It is called
        from train method (maybe several times)
        Args:
            kwargs should include all entries defined in self._pupil_default_training"""
        # print("(Environment._train)self._hooks:", self._hooks)
        # print("(Environment._train)cwd:", os.getcwd())
        train_specs = construct(run_specs['train_specs'])
        # print("(Environment._train)train_specs['train_batch_kwargs']:", train_specs['train_batch_kwargs'])
        schedule = construct(run_specs['schedule'])
        step = init_step

        storage = self._current_place_for_result_saving
        # creating batch generator

        # resetting step in control_storage
        storage['step'] = step
        # self.set_in_storage(step=step)
        train_feed_dict_additions = train_specs['additions_to_feed_dict']
        # print("(Environment._train)train_feed_dict_additions:", train_feed_dict_additions)
        validation_additional_feed_dict = train_specs['validation_additions_to_feed_dict']
        stop_specs = construct(train_specs['stop'])

        # print('train_feed_dict_additions:', train_feed_dict_additions)
        additional_controllers = list()
        for addition in train_feed_dict_additions:
            # print("(Environment._train)addition:", addition)
            additional_controllers.append(Controller(storage, addition['value']))

        to_be_collected_while_training = schedule['to_be_collected_while_training']
        collect_interval = to_be_collected_while_training['results_collect_interval']
        print_per_collected = to_be_collected_while_training['print_per_collected']
        example_per_print = to_be_collected_while_training['example_per_print']

        if train_specs['no_validation'] or collect_interval is None:
            it_is_time_for_validation = Controller(storage,
                                                   {'type': 'always_false'})
            it_is_time_for_example = Controller(storage,
                                                {'type': 'always_false'})
        else:
            valid_period = collect_interval * print_per_collected
            it_is_time_for_validation = Controller(storage,
                                                   {'type': 'periodic_truth',
                                                    'period': valid_period})
            if example_per_print is None:
                it_is_time_for_example = Controller(storage,
                                                    {'type': 'always_false'})
            else:
                example_period = valid_period * example_per_print
                it_is_time_for_example = Controller(storage,
                                                    {'type': 'periodic_truth',
                                                     'period': example_period})

        batch_size_controller = Controller(storage, train_specs['batch_size'])
        batch_size_change_tracker_specs = Controller.create_change_tracking_specifications(train_specs['batch_size'])
        batch_size_should_change = Controller(storage, batch_size_change_tracker_specs)

        if train_specs['debug'] is not None:
            should_start_debugging = Controller(storage, train_specs['debug'])
        else:
            should_start_debugging = Controller(storage,
                                                {'type': 'true_on_steps',
                                                 'steps': []})

        train_batch_kwargs = dict()
        train_batch_kwargs_controller_specs = list()
        for key, batch_arg in train_specs['train_batch_kwargs'].items():
            if isinstance(batch_arg, dict):
                if 'type' in batch_arg:
                    train_batch_kwargs[key] = Controller(storage, batch_arg)
                    train_batch_kwargs_controller_specs.append(batch_arg)
                else:
                    train_batch_kwargs[key] = batch_arg
            else:
                train_batch_kwargs[key] = batch_arg
        change_tracker_specs = Controller.create_change_tracking_specifications(
            train_batch_kwargs_controller_specs)
        batch_generator_specs_should_change = Controller(storage, change_tracker_specs)

        # print("(Environment._train)schedule:", schedule)
        # print(
        #     "(Environment._train/before new schedule)self._current_place_for_result_saving",
        #     self._current_place_for_result_saving
        # )
        # print("(Environment._train)[dataset[1] for dataset in train_specs['validation_datasets']:",
        #       [dataset[1] for dataset in train_specs['validation_datasets']])
        # print("(Environment._train)train_specs['validation_datasets']:", train_specs['validation_datasets'])
        self._handler.set_new_run_schedule(
            schedule,
            [dataset[1] for dataset in train_specs['validation_datasets']]
        )
        # print("(Environment._train)cwd:", os.getcwd())
        if checkpoints_path is not None:
            if train_specs['checkpoint_steps'] is not None:
                if train_specs['checkpoint_steps']['type'] == 'true_on_steps':
                    for idx in range(len(train_specs['checkpoint_steps']['steps'])):
                        train_specs['checkpoint_steps']['steps'][idx] += init_step
                it_is_time_to_create_checkpoint = Controller(storage, train_specs['checkpoint_steps'])
            else:
                it_is_time_to_create_checkpoint = Controller(
                    storage,
                    {'type': 'always_false'}
                )
            storage_keys = list(storage.keys())
            if stop_specs['type'] == 'while_progress' or len(storage_keys) > 2:
                if stop_specs['type'] == 'while_progress':
                    path_to_target_metric_storage = stop_specs['path_to_target_metric_storage']
                else:
                    storage_keys.remove('train')
                    storage_keys.remove('step')
                    path_to_target_metric_storage = (storage_keys[0], 'loss')
                best_checkpoint_controller_specs = dict(
                    type='fire_at_best',
                    path_to_target_metric_storage=path_to_target_metric_storage
                )
                # print("(Environment._train)storage:", storage)
                # print("(Environment._train)best_checkpoint_controller_specs:", best_checkpoint_controller_specs)
                it_is_time_to_create_best_checkpoint = Controller(
                    storage,
                    best_checkpoint_controller_specs
                )
            else:
                it_is_time_to_create_best_checkpoint = Controller(
                    storage,
                    {'type': 'always_false'}
                )
        else:
            it_is_time_to_create_checkpoint = Controller(
                storage,
                {'type': 'always_false'}
            )
            it_is_time_to_create_best_checkpoint = Controller(
                storage,
                {'type': 'always_false'}
            )

        # print("(Environment._train)train_specs['learning_rate']:", train_specs['learning_rate'])
        # print(
        #     "(Environment._train/after new schedule)self._current_place_for_result_saving",
        #     self._current_place_for_result_saving
        # )
        if not with_meta_optimizer:
            # print("(Environment._train)storage:", storage)
            learning_rate_controller = Controller(storage,
                                                  train_specs['learning_rate'])
            controllers = [learning_rate_controller]
        else:
            controllers = list()
        # print("(Environment._train)cwd:", os.getcwd())
        controllers.extend(additional_controllers)
        controllers.append(batch_size_controller)
        batch_kwargs_controllers = list()
        for batch_kwarg in train_batch_kwargs.values():
            if isinstance(batch_kwarg, Controller):
                batch_kwargs_controllers.append(batch_kwarg)
        controllers.extend(batch_kwargs_controllers)
        self._handler.set_controllers(controllers)

        batch_size = batch_size_controller.get()
        tb_kwargs = self._build_batch_kwargs(train_batch_kwargs)
        # print("(Environment._train)tb_kwargs:", tb_kwargs)
        train_batches = batch_generator_class(train_specs['train_dataset'][0], batch_size, **tb_kwargs)
        feed_dict = dict()
        # print("(Environment._train)cwd:", os.getcwd())
        if stop_specs['type'] == 'limit_steps':
            stop_specs['limit'] += init_step
        elif stop_specs['type'] == 'while_progress':
            # print("(Environment._train)stop_specs['changing_parameter_name']:", stop_specs['changing_parameter_name'])
            if stop_specs['changing_parameter_name'] == 'learning_rate':
                stop_specs['changing_parameter_controller'] = learning_rate_controller             
        should_continue = Controller(storage, stop_specs)
        while should_continue.get():
            # print("(Environment._train)cwd:", os.getcwd())
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
                # print("(Environment._train)self._hooks:", self._hooks)
                # print("(Environment._train)addition['placeholder']:", addition['placeholder'])
                feed_dict[self._hooks[addition['placeholder']]] = add_controller.get()
            # print('(Environment._train)self._hooks:', self._hooks)

            train_operations = self._handler.get_tensors('train', step, with_meta_optimizer=with_meta_optimizer)
            # print('train_operations:', train_operations)
            # print('feed_dict:', feed_dict)

            train_res = self._session.run(train_operations, feed_dict=feed_dict)
            # here loss is given in bits per input (BPI)
            self._handler.process_results(step, train_res, regime='train')
            # print("(Environment._train)train_specs['valid_batch_kwargs']:", train_specs['valid_batch_kwargs'])
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
            if it_is_time_to_create_best_checkpoint.get():
                self._create_checkpoint('best', checkpoints_path)
            step += 1
            storage['step'] = step
        return step

    def train(
            self,
            *args,
            start_session=True,
            close_session=True,
            set_passed_parameters_as_default=False,
            **kwargs
    ):
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
        tmp_output = parse_train_method_arguments(
            self,
            args,
            kwargs,
            set_passed_parameters_as_default=set_passed_parameters_as_default
        )
        session_specs = tmp_output['session_specs']
        start_specs = tmp_output['start_specs']
        run_specs_set = tmp_output['run']
        # print("(Environment.train)run_specs_set[0]['train_specs']['train_batch_kwargs']:",
        #       run_specs_set[0]['train_specs']['train_batch_kwargs'])
        # print('(Environment.train)all_tensor_aliases:', all_tensor_aliases)

        if start_session:
            self._start_session(session_specs['allow_soft_placement'],
                                session_specs['log_device_placement'],
                                session_specs['gpu_memory'],
                                session_specs['allow_growth'],
                                session_specs['visible_device_list'])
        train_time = self._train_repeatedly(start_specs, run_specs_set)
        if close_session:
            self._close_session()
        return train_time

    def _train_repeatedly(self, start_specs, run_specs_set):
        # initializing model
        self.flush_storage()
        self._session.run(tf.global_variables_initializer())
        self._restore_pupil(start_specs['restore_path'])
        if start_specs['with_meta_optimizer']:
            self._restore_meta_optimizer(start_specs['restore_optimizer_path'])
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
        # if 'reset_pupil_train_state' in self._hooks:
        #     self._session.run(self._hooks['reset_pupil_train_state'])
        if checkpoints_path is not None:
            self._create_checkpoint('start', checkpoints_path)
        t1 = time.clock()
        for run_specs in run_specs_set:
            init_step = self._train(run_specs,
                                    checkpoints_path,
                                    start_specs['batch_generator_class'],
                                    start_specs['with_meta_optimizer'],
                                    init_step=init_step)
        train_time = time.clock() - t1
        if checkpoints_path is not None:
            self._create_checkpoint('final', checkpoints_path)
        self._handler.log_finish_time()
        self._handler.close()
        return train_time

    def train_optimizer(
            self,
            *args,
            start_session=True,
            close_session=True,
            set_passed_parameters_as_default=False,
            **kwargs
    ):
        self._store_launch_parameters(
            'optimizer',
            args=args,
            start_session=start_session,
            close_session=close_session,
            set_passed_parameters_as_default=set_passed_parameters_as_default,
            kwargs=kwargs)
        # print("\n\n(Environment.train_optimizer)kwargs:", kwargs)
        tmp_output = parse_train_optimizer_method_arguments(
            self,
            args,
            kwargs,
            set_passed_parameters_as_default=set_passed_parameters_as_default
        )
        # print("\n\n(Environment.train_optimizer)tmp_output:", tmp_output)
        session_specs = tmp_output['session_specs']
        start_specs = tmp_output['start_specs']
        run_specs_set = tmp_output['run']
        # print('(Environment.train_optimizer)all_tensor_aliases:', all_tensor_aliases)
        # print("(Environment.train_optimizer)run_specs_set[0]['optimizer_inference']['valid_batch_kwargs']:",
        #       run_specs_set[0]['optimizer_inference']['valid_batch_kwargs'])

        if start_session:
            self._start_session(session_specs['allow_soft_placement'],
                                session_specs['log_device_placement'],
                                session_specs['gpu_memory'],
                                session_specs['allow_growth'],
                                session_specs['visible_device_list'])
        train_time = self._train_optimizer_repeatedly(start_specs, run_specs_set)
        if close_session:
            self._close_session()
        return train_time

    def _train_optimizer_repeatedly(self, start_specs, run_specs_set, log=True):
        # initializing model
        self.flush_storage()
        # print("(Environment._train_optimizer_repeatedly)all_trainable:")
        # for v in tf.trainable_variables():
        #     print(v)
        # print("(Environment._train_optimizer_repeatedly)global_variables:")
        # for v in tf.global_variables():
        #     print(v)
        self._session.run(tf.global_variables_initializer())
        self._restore_meta_optimizer(start_specs['restore_optimizer_path'])
        processing_type = 'train_meta_optimizer'

        self._handler = Handler(self,
                                self._hooks,
                                processing_type,
                                start_specs['save_path'],
                                start_specs['result_types'],
                                summary=start_specs['summary'],
                                add_graph_to_summary=start_specs['add_graph_to_summary'],
                                batch_generator_class=start_specs['batch_generator_class'],
                                vocabulary=start_specs['vocabulary'])
        if log:
            self._handler.log_launch()
        if start_specs['save_path'] is not None:
            checkpoints_path = start_specs['save_path'] + '/checkpoints'
            create_path(checkpoints_path)
        else:
            checkpoints_path = None
        init_step = 0
        t1 = time.clock()
        for run_specs in run_specs_set:
            # print("(Environment._train_optimizer_repeatedly)"
            #       "run_specs",
            #       run_specs)
            # print("(Environment._train_optimizer_repeatedly)"
            #       "run_specs['optimizer_inference']['opt_inf_train_datasets'][0][1]",
            #       run_specs['optimizer_inference']['opt_inf_train_datasets'][0][1])
            # print("(Environment._train_optimizer_repeatedly)"
            #       "run_specs['optimizer_inference']['opt_inf_validation_datasets'][0][1]",
            #       run_specs['optimizer_inference']['opt_inf_validation_datasets'][0][1])
            init_step = self._train_optimizer(
                run_specs,
                checkpoints_path,
                start_specs['batch_generator_class'],
                start_specs['result_types'],
                init_step=init_step
            )
        train_time = time.clock() - t1
        if checkpoints_path is not None:
            self._create_checkpoint('final', checkpoints_path, model_type='optimizer')
        self._handler.log_finish_time()
        self._handler.close()
        return train_time

    def _fill_train_meta_optimizer_feed_dict_with_inputs_and_labels(
            self,
            feed_dict,
            pupil_grad_eval_batch_gens,
            optimizer_grad_batch_gens,
            share_train_data
    ):
        if share_train_data:
            for pup_inp_plh, pup_lbl_plh, opt_inp_plh, opt_lbl_plh, b_gen in zip(
                    self._hooks['pupil_grad_eval_inputs'],
                    self._hooks['pupil_grad_eval_labels'],
                    self._hooks['optimizer_grad_inputs'],
                    self._hooks['optimizer_grad_labels'],
                    pupil_grad_eval_batch_gens
            ):
                if isinstance(pup_inp_plh, list):
                    for pup_inp_plhld, pup_lbl_plhld, opt_inp_plhld, opt_lbl_plhld in zip(
                            pup_inp_plh,
                            pup_lbl_plh,
                            opt_inp_plh,
                            opt_lbl_plh
                    ):
                        inp, lbl = b_gen.next()
                        feed_dict[pup_inp_plhld] = inp
                        feed_dict[pup_lbl_plhld] = lbl
                        feed_dict[opt_inp_plhld] = inp
                        feed_dict[opt_lbl_plhld] = lbl
                else:
                    inp, lbl = b_gen.next()
                    feed_dict[pup_inp_plh] = inp
                    feed_dict[pup_lbl_plh] = lbl
                    feed_dict[opt_inp_plh] = inp
                    feed_dict[opt_lbl_plh] = lbl
        else:
            for inp_placeholder, lbl_placeholder, b_gen in zip(
                    self._hooks['pupil_grad_eval_inputs'],
                    self._hooks['pupil_grad_eval_labels'],
                    pupil_grad_eval_batch_gens
            ):
                if isinstance(inp_placeholder, list):
                    for idx, (inp_plhld, lbl_plhld) in enumerate(zip(inp_placeholder, lbl_placeholder)):
                        inp, lbl = b_gen.next()
                        # print(
                        #     "\n\npupil grad eval\n"
                        #     "(Environment._fill_train_meta_optimizer_feed_dict_with_inputs_and_labels)inp:\n",
                        #     inp
                        # )
                        # print("(Environment._fill_train_meta_optimizer_feed_dict_with_inputs_and_labels)lbl:\n", lbl)
                        feed_dict[inp_plhld] = inp
                        feed_dict[lbl_plhld] = lbl
                else:
                    inp, lbl = b_gen.next()
                    feed_dict[inp_placeholder] = inp
                    feed_dict[lbl_placeholder] = lbl
            for inp_placeholder, lbl_placeholder, b_gen in zip(
                    self._hooks['optimizer_grad_inputs'],
                    self._hooks['optimizer_grad_labels'],
                    optimizer_grad_batch_gens
            ):
                if isinstance(inp_placeholder, list):
                    for idx, (inp_plhld, lbl_plhld) in enumerate(zip(inp_placeholder, lbl_placeholder)):
                        inp, lbl = b_gen.next()
                        # print(
                        #     "\n\noptimizer grad eval\n"
                        #     "(Environment._fill_train_meta_optimizer_feed_dict_with_inputs_and_labels)inp:\n",
                        #     inp
                        # )
                        # print("(Environment._fill_train_meta_optimizer_feed_dict_with_inputs_and_labels)lbl:\n", lbl)
                        feed_dict[inp_plhld] = inp
                        feed_dict[lbl_plhld] = lbl
                else:
                    inp, lbl = b_gen.next()
                    feed_dict[inp_placeholder] = inp
                    feed_dict[lbl_placeholder] = lbl
        return feed_dict

    def _reset_exercises(
            self,
            num_exercises,
            pupil_restore_paths,
            datasets,
            batch_generator_class,
            batch_size_controller,
            batch_gen_init_is_random,
            train_batch_kwargs,
            restore_paths_datasets_map,
            share_train_data,
            one_batch_gen,
            random_=True
    ):
        # print("EXERCISES RESET!")
        # print('(Environment._reset_exercises)restore_paths_datasets_map:', restore_paths_datasets_map)
        # print('(Environment._reset_exercises)datasets:', datasets)
        num_paths = len(pupil_restore_paths)
        # print("(Environment._reset_exercises)num_paths:", num_paths)
        # print("(Environment._reset_exercises)num_exercises:", num_exercises)
        if random_:
            if num_paths > num_exercises:
                paths = random.sample(list(enumerate(pupil_restore_paths)), num_exercises)
            else:
                map_ = create_distribute_map(num_paths, num_exercises)
                paths = [pupil_restore_paths[i] for i in map_]
            if restore_paths_datasets_map is None:
                restore_paths_datasets_map = [random.choice(
                    [i for i, _ in enumerate(datasets)]) for _ in paths]
        else:
            if num_paths > num_exercises:
                paths = pupil_restore_paths[:num_exercises]
            else:
                map_ = create_distribute_map(num_paths, num_exercises)
                paths = [pupil_restore_paths[i] for i in map_]
            if restore_paths_datasets_map is None:
                restore_paths_datasets_map = create_distribute_map(len(datasets), len(paths))
        pupil_grad_eval_batch_gens = list()
        if share_train_data:
            optimizer_grad_batch_gens = None
        else:
            optimizer_grad_batch_gens = list()
        batch_size = batch_size_controller.get()
        tb_kwargs = self._build_batch_kwargs(train_batch_kwargs)
        # print("(Environment._reset_exercises)tb_kwargs:", tb_kwargs)
        # print("(Environment._reset_exercises)restore_paths_datasets_map:", restore_paths_datasets_map)
        # print("(Environment._reset_exercises)datasets:", datasets)
        if one_batch_gen:
            bg = batch_generator_class(
                datasets[restore_paths_datasets_map[0]][0],
                batch_size,
                **tb_kwargs,
                random_batch_initiation=batch_gen_init_is_random
            )
        for idx, (saver, pupil_trainable_initializer, path) in enumerate(
                zip(self._hooks['pupil_savers'], self._hooks['pupil_trainable_initializers'], paths)):
            if path is None:
                self._session.run(pupil_trainable_initializer)
            else:
                # print("(Environmet._reset_exercises)path:", path)
                saver.restore(self._session, path)
            # print("(Environment._reset_exercises)restore_paths_datasets_map:", restore_paths_datasets_map)
            # print("(Environment._reset_exercises)idx:", idx)
            if not one_batch_gen:
                bg = batch_generator_class(
                    datasets[restore_paths_datasets_map[idx]][0],
                    batch_size,
                    **tb_kwargs,
                    random_batch_initiation=batch_gen_init_is_random
                )
            pupil_grad_eval_batch_gens.append(bg)
            if not share_train_data:
                if not one_batch_gen:
                    bg = batch_generator_class(
                        datasets[restore_paths_datasets_map[idx]][0],
                        batch_size,
                        **tb_kwargs,
                        random_batch_initiation=batch_gen_init_is_random
                    )
                optimizer_grad_batch_gens.append(bg)
        # print("(Environment._reset_exercises)len(pupil_grad_eval_batch_gens):", len(pupil_grad_eval_batch_gens))
        # print(
        #     "(Environment._reset_exercises)len(self._hooks['pupil_trainable_initializers']):",
        #     len(self._hooks['pupil_trainable_initializers'])
        # )
        # print("(Environment._reset_exercises)len(self._hooks['pupil_savers']):", len(self._hooks['pupil_savers']))
        self._session.run(self._hooks['reset_permutation_matrices'])
        self._session.run(self._hooks['reset_optimizer_train_state'])
        self._session.run(self._hooks['reset_pupil_grad_eval_pupil_storage'])
        self._session.run(self._hooks['reset_optimizer_grad_pupil_storage'])
        return pupil_grad_eval_batch_gens, optimizer_grad_batch_gens

    @staticmethod
    def _create_train_method_run_specs_from_meta_optimizer_train_method_arguments(
            train_specs,
            optimizer_inference,
            schedule,
            train_dataset,
            validation_dataset
    ):
        # print("(Environment._create_train_method_run_specs_from_meta_optimizer_train_method_arguments)"
        #       "train_specs['train_batch_kwargs']:", train_specs['train_batch_kwargs'])
        # print("(Environment._create_train_method_run_specs_from_meta_optimizer_train_method_arguments)"
        #       "optimizer_inference['valid_batch_kwargs']:", optimizer_inference['valid_batch_kwargs'])
        new_train_specs = dict(
            learning_rate=None,
            additions_to_feed_dict=optimizer_inference['opt_inf_additions_to_feed_dict'],
            stop=optimizer_inference['opt_inf_stop'],
            train_dataset=train_dataset,
            batch_size=train_specs['batch_size'],
            train_batch_kwargs=train_specs['train_batch_kwargs'],
            checkpoint_steps=None,
            debug=None,
            validation_datasets=[validation_dataset],
            validation_additions_to_feed_dict=optimizer_inference['validation_additions_to_feed_dict'],
            validation_batch_size=optimizer_inference['validation_batch_size'],
            valid_batch_kwargs=optimizer_inference['valid_batch_kwargs'],
            validate_tokens_by_chars=optimizer_inference['validate_tokens_by_chars'],
            no_validation=optimizer_inference['no_validation']
        )
        collected_while_training = dict(
            results_collect_interval=optimizer_inference[
                'opt_inf_to_be_collected_while_training']['opt_inf_results_collect_interval'],
            print_per_collected=optimizer_inference[
                'opt_inf_to_be_collected_while_training']['opt_inf_print_per_collected'],
            example_per_print=optimizer_inference[
                'opt_inf_to_be_collected_while_training']['opt_inf_example_per_print']
        )
        new_schedule = dict(
            to_be_collected_while_training=collected_while_training,
            printed_result_types=schedule['printed_result_types'],
            printed_controllers=schedule['printed_controllers'],
            fuses=optimizer_inference['fuses'],
            fuse_tensors=optimizer_inference['fuse_tensors'],
            example_length=optimizer_inference['example_length'],
            example_tensors=optimizer_inference['example_tensors'],
            replicas=optimizer_inference['replicas'],
            random=optimizer_inference['random'],
            train_tensor_schedule=optimizer_inference['opt_inf_train_tensor_schedule'],
            validation_tensor_schedule=optimizer_inference['opt_inf_validation_tensor_schedule']
        )
        return {'train_specs': new_train_specs, 'schedule': new_schedule}

    def _launch_optimizer_inference(
            self,
            train_specs,
            optimizer_inference,
            schedule,
            pupil_idx,
            pupil_name,
            pupil_path,
            optimizer_training_step,
            batch_generator_class,
            result_types,
            storage=None
    ):
        # print('\nOptimizer inference on pupil "%s"' % pupil_name)
        # print("(Environment._train_optimizer)optimizer_inference['opt_inf_train_datasets']:",
        #       optimizer_inference['opt_inf_train_datasets'])
        # print("(Environment._train_optimizer)optimizer_inference['opt_inf_validation_datasets']:",
        #       optimizer_inference['opt_inf_validation_datasets'])

        # print("(Environment._launch_optimizer_inference)optimizer_inference['opt_inf_train_datasets'][pupil_idx][1]:",
        #       optimizer_inference['opt_inf_train_datasets'][pupil_idx][1])
        train_dataset = optimizer_inference['opt_inf_train_datasets'][pupil_idx]
        validation_dataset = optimizer_inference['opt_inf_validation_datasets'][pupil_idx]
        train_dataset[1] = 'train'
        train_dataset_name = train_dataset[1]
        validation_dataset[1] = 'validation'
        validation_dataset_name = validation_dataset[1]
        run_specs = self._create_train_method_run_specs_from_meta_optimizer_train_method_arguments(
            train_specs,
            optimizer_inference,
            schedule,
            train_dataset,
            validation_dataset
        )
        # print("(Environment._train_optimizer)self._hooks:", self._hooks)
        if 'reset_pupil_train_state' in self._hooks:
            self._session.run(self._hooks['reset_pupil_train_state'])
        self._session.run(self._hooks['reset_optimizer_inference_state'])
        print('OPTIMIZER INFERENCE ON PUPIL %s\n' % pupil_name + '*' * 40)
        self._restore_pupil(pupil_path)
        self._handler.set_pupil_name(pupil_name)
        self._handler.set_meta_optimizer_training_step(optimizer_training_step)
        self._handler.set_meta_optimizer_inference_flags(True, False)

        if storage is None:
            storage = dict()
            self._current_place_for_result_saving[pupil_name]['results'].append(storage)
        old_place_for_saving = self._current_place_for_result_saving
        dataset_names = [train_dataset_name, validation_dataset_name]
        # print("(Environment._launch_optimizer_inference)dataset_names:", dataset_names)
        self._current_place_for_result_saving = self.create_train_pupil_storage(
            storage, result_types, dataset_names
        )
        # print("(Environment._launch_optimizer_inference)self._current_place_for_result_saving:",
        #       self._current_place_for_result_saving)
        _ = self._train(
            run_specs,
            None,
            batch_generator_class,
            True,
            init_step=0,
        )
        print('*' * 40)
        self._handler.set_pupil_name(None)
        self._handler.set_meta_optimizer_training_step(None)
        self._handler.set_meta_optimizer_inference_flags(False, False)
        self._current_place_for_result_saving = old_place_for_saving

    def _train_optimizer(
            self,
            run_specs,
            checkpoints_path,
            batch_generator_class,
            result_types,
            init_step=0
    ):
        """It is a method that does actual training and responsible for one training pass through dataset. It is called
        from train method (maybe several times)
        Args:
            kwargs should include all entries defined in self._pupil_default_training"""
        train_specs = construct(run_specs['train_specs'])
        schedule = construct(run_specs['schedule'])
        optimizer_inference = construct(run_specs['optimizer_inference'])
        step = init_step
        # print("(Environment._train_optimizer)optimizer_inference:", optimizer_inference)
        if optimizer_inference['opt_inf_pupil_restore_paths'] is None:
            opt_inf_pupil_names = None
        else:
            opt_inf_pupil_names = nth_element_of_sequence_of_sequences(
                optimizer_inference['opt_inf_pupil_restore_paths'],
                0
            )
        self._handler.set_optimizer_train_schedule(
            schedule,
            opt_inf_pupil_names=opt_inf_pupil_names,
            opt_inf_to_be_collected_while_training=optimizer_inference['opt_inf_to_be_collected_while_training'],
            opt_inf_train_tensor_schedule=optimizer_inference['opt_inf_train_tensor_schedule'],
            opt_inf_validation_tensor_schedule=optimizer_inference['opt_inf_validation_tensor_schedule']
        )

        # creating batch generator

        # resetting step in control_storage
        self.set_in_storage(step=step)
        learning_rate_controller = Controller(self._current_place_for_result_saving,
                                              train_specs['learning_rate'])
        train_feed_dict_additions = train_specs['additions_to_feed_dict']

        # print('train_feed_dict_additions:', train_feed_dict_additions)
        additional_controllers = list()
        for addition in train_feed_dict_additions:
            # print("(Environment._train_optimizer)addition:", addition)
            additional_controllers.append(Controller(self._current_place_for_result_saving, addition['value']))
        # print("(Environment._train_optimizer)additional_controllers:", [ac.name for ac in additional_controllers])
        if train_specs['stop']['type'] == 'limit_steps':
            train_specs['stop']['limit'] += init_step
        should_continue = Controller(self._current_place_for_result_saving, train_specs['stop'])

        to_be_collected_while_training = schedule['to_be_collected_while_training']
        collect_interval = to_be_collected_while_training['results_collect_interval']
        print_per_collected = to_be_collected_while_training['print_per_collected']

        if optimizer_inference['opt_inf_is_performed']:
            opt_inf_period = collect_interval * print_per_collected
            it_is_time_for_opt_inf = Controller(
                self._current_place_for_result_saving,
                {'type': 'periodic_truth',
                 'period': opt_inf_period})

        else:
            it_is_time_for_opt_inf = Controller(
                self._current_place_for_result_saving,
                {'type': 'always_false'}
            )
        batch_size_controller = Controller(self._current_place_for_result_saving, train_specs['batch_size'])
        if train_specs['checkpoint_steps'] is not None and checkpoints_path is not None:
            if train_specs['checkpoint_steps']['type'] == 'true_on_steps':
                for idx in range(len(train_specs['checkpoint_steps']['steps'])):
                    train_specs['checkpoint_steps']['steps'][idx] += init_step
            it_is_time_to_create_checkpoint = Controller(
                self._current_place_for_result_saving, train_specs['checkpoint_steps'])
        else:
            it_is_time_to_create_checkpoint = Controller(self._current_place_for_result_saving,
                                                         {'type': 'always_false'})

        # print("(Environment._train_optimizer)train_specs['reset_period']:", train_specs['reset_period'])
        it_is_time_to_reset_exercises = Controller(self._current_place_for_result_saving, train_specs['reset_period'])

        if train_specs['debug'] is not None:
            should_start_debugging = Controller(self._current_place_for_result_saving, train_specs['debug'])
        else:
            should_start_debugging = Controller(self._current_place_for_result_saving,
                                                {'type': 'true_on_steps',
                                                 'steps': []})

        train_batch_kwargs = dict()
        train_batch_kwargs_controller_specs = list()
        for key, batch_arg in train_specs['train_batch_kwargs'].items():
            if isinstance(batch_arg, dict):
                if 'type' in batch_arg:
                    train_batch_kwargs[key] = Controller(self._current_place_for_result_saving, batch_arg)
                    train_batch_kwargs_controller_specs.append(batch_arg)
                else:
                    train_batch_kwargs[key] = batch_arg
            else:
                train_batch_kwargs[key] = batch_arg

        if 'learning_rate' in schedule['printed_controllers']:
            controllers_for_printing = [learning_rate_controller]
        else:
            controllers_for_printing = []

        controllers_for_printing.extend(additional_controllers)

        self._handler.set_controllers(controllers_for_printing)
        # print("(Environment._train_optimizer)train_specs['train_datasets']:", train_specs['train_datasets'])
        # print("(Environment._train_optimizer)train_specs['num_exercises']:", train_specs['num_exercises'])
        pupil_grad_eval_batch_gens, optimizer_grad_batch_gens = self._reset_exercises(
            train_specs['num_exercises'],
            train_specs['pupil_restore_paths'],
            train_specs['train_datasets'],
            batch_generator_class,
            batch_size_controller,
            train_specs['batch_gen_init_is_random'],
            train_batch_kwargs,
            train_specs['restore_paths_datasets_map'],
            train_specs['share_train_data'],
            train_specs['one_batch_gen'],
            random_=False
        )
        feed_dict = dict()
        # print("(Environment._train_optimizer)before loop")

        while should_continue.get():
            if should_start_debugging.get():
                self._session = tf_debug.LocalCLIDebugWrapperSession(self._session)
                self._session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            if it_is_time_to_create_checkpoint.get():
                self._create_checkpoint(step, checkpoints_path, model_type='optimizer')

            feed_dict = self._fill_train_meta_optimizer_feed_dict_with_inputs_and_labels(
                feed_dict, pupil_grad_eval_batch_gens, optimizer_grad_batch_gens, train_specs['share_train_data'])

            learning_rate = learning_rate_controller.get()
            feed_dict[self._hooks['learning_rate_for_optimizer_training']] = learning_rate
            for addition, add_controller in zip(train_feed_dict_additions, additional_controllers):
                feed_dict[self._hooks[addition['placeholder']]] = add_controller.get()
            # print('(Environment._train)self._hooks:', self._hooks)
            train_operations = self._handler.get_tensors('train_meta_optimizer', step)
            # print('train_operations:', train_operations)
            # print('feed_dict:', feed_dict)
            # print("(Environment._train_optimizer)feed_dict:", feed_dict)
            train_res = self._session.run(train_operations, feed_dict=feed_dict)
            # here loss is given in bits per input (BPI)
            # print("(Environment._train_optimizer)after session run")
            self._handler.process_results(step, train_res, regime='train_meta_optimizer')
            if it_is_time_for_opt_inf.get():
                for idx, (pupil_name, path) in enumerate(optimizer_inference['opt_inf_pupil_restore_paths']):
                    self._launch_optimizer_inference(
                        train_specs,
                        optimizer_inference,
                        schedule,
                        idx,
                        pupil_name,
                        path,
                        step,
                        batch_generator_class,
                        result_types
                    )

            step += 1
            if it_is_time_to_reset_exercises.get():
                pupil_grad_eval_batch_gens, optimizer_grad_batch_gens = self._reset_exercises(
                    train_specs['num_exercises'],
                    train_specs['pupil_restore_paths'],
                    train_specs['train_datasets'],
                    batch_generator_class,
                    batch_size_controller,
                    train_specs['batch_gen_init_is_random'],
                    train_batch_kwargs,
                    train_specs['restore_paths_datasets_map'],
                    train_specs['share_train_data'],
                    train_specs['one_batch_gen'],
                )
            self.set_in_storage(step=step)
        return step

    def _preparations_for_launch(
            self,
            kwargs_for_building,
            session_specs,
            evaluation,
            meta_optimizer_build_kwargs
    ):
        tf.set_random_seed(1)
        self._build_pupil(kwargs_for_building)
        if meta_optimizer_build_kwargs is not None:
            self.build_optimizer(**meta_optimizer_build_kwargs)
        # print('args_for_launches:', args_for_launches)
        self._start_session(session_specs['allow_soft_placement'],
                            session_specs['log_device_placement'],
                            session_specs['gpu_memory'],
                            session_specs['allow_growth'],
                            session_specs['visible_device_list'])
        datasets = dict([(d[1], d) for d in evaluation['datasets']])
        if 'train' in datasets:
            del datasets['train']
        if evaluation['batch_gen_class'] is None:
            eval_batch_gen_class = self._default_batch_generator
        else:
            eval_batch_gen_class = evaluation['batch_gen_class']

        additional_feed_dict = self._form_validation_additional_feed_dict([], [], evaluation['additional_feed_dict'])
        return additional_feed_dict, eval_batch_gen_class, datasets

    def _launch_and_put_in_queue(
            self,
            hp_comb,
            order,
            start_specs,
            run_specs_set,
            evaluation,
            datasets,
            eval_batch_gen_class,
            additional_feed_dict,
            queue_
    ):
        self._handler.print_hyper_parameters(hp_comb, order)
        result = dict()
        self._train_repeatedly(start_specs, run_specs_set)
        if 'train' in evaluation['datasets']:
            tr_res = dict()
            for key, res in self._storage['train'].items():
                if len(res) > 0:
                    tr_res[key] = res[-1]
            result['train'] = tr_res
        validation_dataset_names = [d[1] for d in evaluation['datasets']]
        self._handler = Handler(
            self,
            self._hooks,
            'test',
            None,
            evaluation['result_types'],
            validation_dataset_names=validation_dataset_names
        )
        for dataset_name, dataset in datasets.items():
            # print('dataset_name:', dataset_name)
            # print('dataset:', dataset)
            means = self._validate(
                eval_batch_gen_class,
                dataset,
                evaluation['batch_size'],
                evaluation['batch_kwargs'],
                additional_feed_dict=additional_feed_dict,
                save_to_file=False,
                save_to_storage=False,
                print_results=False
            )
            result[dataset_name] = means
        # print('result in process:', result)
        queue_.put(result)

    def _several_launches_without_rebuilding(
            self,
            queue_,
            kwargs_for_building,
            session_specs,
            args_for_launches,
            evaluation,
            hp_combs,
            order,
            meta_optimizer_build_kwargs
    ):

        additional_feed_dict, eval_batch_gen_class, datasets = self._preparations_for_launch(
            kwargs_for_building,
            session_specs,
            evaluation,
            meta_optimizer_build_kwargs
        )
        for hp_comb, (start_specs, run_specs_set) in zip(hp_combs, args_for_launches):
            self._launch_and_put_in_queue(
                hp_comb,
                order,
                start_specs,
                run_specs_set,
                evaluation,
                datasets,
                eval_batch_gen_class,
                additional_feed_dict,
                queue_
            )

    def _one_launch(
            self,
            queue_,
            kwargs_for_building,
            session_specs,
            start_specs,
            run_specs_set,
            evaluation,
            hp_comb,
            order,
            meta_optimizer_build_kwargs
    ):
        additional_feed_dict, eval_batch_gen_class, datasets = self._preparations_for_launch(
            kwargs_for_building,
            session_specs,
            evaluation,
            meta_optimizer_build_kwargs
        )
        self._launch_and_put_in_queue(
            hp_comb,
            order,
            start_specs,
            run_specs_set,
            evaluation,
            datasets,
            eval_batch_gen_class,
            additional_feed_dict,
            queue_
        )

    def _launch_optimizer_and_put_in_queue(
            self,
            hp_comb,
            order,
            start_specs,
            run_specs_set,
            evaluation,
            queue_
    ):
        # print("(Environment._several_optimizer_launches_without_rebuilding)run_specs_set:", run_specs_set)
        # print("(Environment._several_optimizer_launches_without_rebuilding)run_specs_set[0]['train_specs']:",
        #       run_specs_set[0]['train_specs'])
        # print("(Environment._several_optimizer_launches_without_rebuilding)run_specs_set")
        # for idx, rss in enumerate(run_specs_set):
        #     print(idx)
        #     for k, v in rss.items():
        #         print(k)
        #         print(v)
        self._handler.print_hyper_parameters(hp_comb, order)

        result_types = start_specs['result_types']
        self._train_optimizer_repeatedly(start_specs, run_specs_set, log=False)
        pupil_names = nth_element_of_sequence_of_sequences(
            evaluation['opt_inf_pupil_restore_paths'],
            0
        )
        self._handler.set_optimizer_train_schedule(
            None,
            opt_inf_pupil_names=pupil_names,
            opt_inf_to_be_collected_while_training=evaluation['opt_inf_to_be_collected_while_training']
        )
        result = dict()
        # print("(Environment._several_optimizer_launches_without_rebuilding)pupil_names:", pupil_names)
        for idx, name in enumerate(pupil_names):
            train_dataset_name = evaluation['opt_inf_train_datasets'][idx][1]
            # duplicates validation storage setting in Handler.set_new_run_schedule
            if evaluation['opt_inf_validation_datasets'] is not None:
                validation_dataset_name = 'validation'
            else:
                validation_dataset_name = None
            dataset_names = ['train']
            if validation_dataset_name is not None:
                dataset_names.append(validation_dataset_name)
            result[name] = self.create_train_pupil_storage(
                dict(), result_types, dataset_names
            )

        old_place_for_saving = self._current_place_for_result_saving
        self._current_place_for_result_saving = result
        for idx, (pupil_name, pupil_path) in enumerate(evaluation['opt_inf_pupil_restore_paths']):
            self._launch_optimizer_inference(
                run_specs_set[0]['train_specs'],
                evaluation,
                run_specs_set[0]['schedule'],
                idx,
                pupil_name,
                pupil_path,
                0,
                start_specs['batch_generator_class'],
                result_types,
                storage=result[pupil_name]
            )
        self._current_place_for_result_saving = old_place_for_saving
        queue_.put(result)

    def _several_optimizer_launches_without_rebuilding(
            self,
            queue_,
            pupil_build_kwargs,
            optimizer_build_kwargs,
            session_specs,
            args_for_launches,
            evaluation,
            hp_combs,
            order
    ):
        self._build_pupil(pupil_build_kwargs)
        self.build_optimizer(**optimizer_build_kwargs)
        self._start_session(session_specs['allow_soft_placement'],
                            session_specs['log_device_placement'],
                            session_specs['gpu_memory'],
                            session_specs['allow_growth'],
                            session_specs['visible_device_list'])
        for hp_comb, (start_specs, run_specs_set) in zip(hp_combs, args_for_launches):
            self._launch_optimizer_and_put_in_queue(
                hp_comb,
                order,
                start_specs,
                run_specs_set,
                evaluation,
                queue_
            )

    def _one_optimizer_launch(
            self,
            queue_,
            pupil_build_kwargs,
            optimizer_build_kwargs,
            session_specs,
            start_specs,
            run_specs_set,
            evaluation,
            hp_comb,
            order
    ):
        self._build_pupil(pupil_build_kwargs)
        self.build_optimizer(**optimizer_build_kwargs)
        self._start_session(session_specs['allow_soft_placement'],
                            session_specs['log_device_placement'],
                            session_specs['gpu_memory'],
                            session_specs['allow_growth'],
                            session_specs['visible_device_list'])
        # print("(Environment._one_optimizer_launch)run_specs_set:", run_specs_set)
        self._launch_optimizer_and_put_in_queue(
            hp_comb,
            order,
            start_specs,
            run_specs_set,
            evaluation,
            queue_
        )

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

    def _spring_process_for_grid_search(
            self,
            args_for_launches,
            shares,
            kwargs_for_building,
            session_specs,
            evaluation,
            other_hp_combs,
            build_hp_comb,
            meta_optimizer_build_kwargs=None,
            rebuild_every_time=True
    ):
        parsed = configure_args_for_launches(self, args_for_launches, shares, model='pupil')
        self.mp_debug_flag += 1

        # constructing all hp combinations with one build hp combination
        hp_combs = list()
        # print("(Environment._spring_process_for_meta_grid_search)other_hp_combs:", other_hp_combs)
        # print("(Environment._spring_process_for_meta_grid_search)build_hp_comb:", build_hp_comb)
        if len(other_hp_combs) > 0:
            for idx, other_hp_comb in enumerate(other_hp_combs):
                hp_combination = construct(build_hp_comb)
                hp_combination.update(other_hp_comb)
                hp_combs.append(hp_combination)
        else:
            hp_combination = construct(build_hp_comb)
            hp_combs.append(hp_combination)
        order = self._handler.order
        if rebuild_every_time:
            for hp_comb, (start_specs, run_specs_set) in zip(hp_combs, parsed):
                # print("(Environment._spring_process_for_meta_grid_search)hp_comb:", hp_comb)
                queue_ = mp.Queue()
                p = mp.Process(
                    target=self._one_launch,
                    args=(
                        queue_,
                        kwargs_for_building,
                        session_specs,
                        start_specs,
                        run_specs_set,
                        evaluation,
                        hp_comb,
                        order,
                        meta_optimizer_build_kwargs,
                    )
                )
                p.start()
                res = queue_.get()
                # print('\nidx: %s\nres: %s' % (idx, res))
                # print('hp_combination:', hp_combination)
                # print("(Environment._spring_process_for_meta_grid_search)res:", res)
                self._handler.process_results(
                    hp_comb, res, regime='several_launches'
                )
                p.join()
        else:
            queue_ = mp.Queue()
            p = mp.Process(
                target=self._several_launches_without_rebuilding,
                args=(
                    queue_,
                    kwargs_for_building,
                    session_specs,
                    args_for_launches,
                    evaluation,
                    hp_combs,
                    order,
                    meta_optimizer_build_kwargs,
                )
            )
            p.start()
            for hp_comb in hp_combs:
                res = queue_.get()
                self._handler.process_results(
                    hp_comb, res, regime='several_launches'
                )
            p.join()

    def grid_search(self,
                    evaluation,
                    kwargs_for_building,
                    build_hyperparameters=None,
                    other_hyperparameters=None,
                    meta_optimizer_build_kwargs=None,
                    initial_experiment_counter_value=0,
                    **kwargs):
        """build_hyperparameters and other_hyperparameters are provided in the following format
        build_hyperparameters and other_hyperparameters are a dictionaries which keys are kwargs for build or train
        Environment methods for corresponding hyper parameters and values are dictionaries of following format:
            hp_type ('build_hp', 'built-in', 'additional_placeholder', 'batch_kwarg')
                can be omitted for build_hyperparameters and if omitted in other_hyperparameters it is set to
                additional_placeholder
            list_indices
                a list of indices of hp values if hp is a list (e.g. number of nodes by layers)
                it can be an int if only 1 index is used
                default is None
            controller
                train kwargs specify parameters such as learning rate which may change during training.
                controller entry is boolean and shows if controller is used
            share
                some parameters are shared between graph building and train methods (e. g. num_unrollings)
                this entry is used in pupil build hps for specifying where to put hp in train method kwargs if
                such procedure is needed. For other_hyperparameters it is set to None.
                share is a dict with 2 entries: 'direction' and 'controller'.
                    'direction' can take values 'built-in', 'additional_placeholder', 'batch_kwarg'
                    'controller' is boolean
            type
                in case when controller is True type is Controller type. If controller is true and type is omitted
                type is set to 'fixed
            fixed
                if parameter is inside dictionary it is likely that dictionary will have another entries. In such case
                fixed is to be used for passing them to model. fixed is a dictionary which entries are entries of
                hp dictionary entries. default is None
            varying
                is an entry for hyper parameter values to be tested.
                varying can be a dictionary which keys are names of specs to be tested and values are list of tested hp
                values. This format is used in cases when hp is a dictionary and and one of dictionary specs is varied,
                e. g. learning_rate
                    {'init': [1., 2., 3.]}
                if hyper parameter is not a dictionary varying is a list of values to be tested
                Note that if list_indices is not None varying contains not list but list of lists where each inner list
                corresponds to list index
        Abbreviations of this format are possible.
            1. you may omit list_indices, hp_type, controller, type, share
            2. for build_hyperparameters you can use following formats
                {'build_hyper_param' = values_to_be_tested}
                {'build_hyper_param[idx1,idx2,idx3]' = [values_list1, values_list2, values_list3]}
                where idx1 and so on are list_indices
        evaluation is an argument specifying how learned models are being tested. evaluation is a dictionary with
        entries:
            datasets: datasets on which model has to be validated
            batch_gen_class: batch generator class used for batch generation
            additional_feed_dict:
            result_types: list containing some of the following: 'loss', 'bpc', 'accuracy', 'perplexity'
            batch_size
            batch_kwargs: additional kwargs for batch generator class initialization
            save_path: path to directory where test results are stored
        """

        self._store_launch_parameters(
            'pupil',
            evaluation=evaluation,
            kwargs_for_building=kwargs_for_building,
            build_hyperparameters=build_hyperparameters,
            other_hyperparameters=other_hyperparameters,
            initial_experiment_counter_value=0,
            kwargs=kwargs)
        if build_hyperparameters is None:
            build_hyperparameters = dict()
        if other_hyperparameters is None:
            other_hyperparameters = dict()
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
        # print("('Environment.grid_search')other_insertions:", other_insertions)
        # print("('Environment.grid_search')build_hp_combs:", build_hp_combs)

        args_for_launches = create_all_args_for_launches(kwargs, other_insertions)
        # print("('Environment.grid_search')args_for_launches[0]['additions_to_feed_dict']:",
        #       args_for_launches[0]['additions_to_feed_dict'])

        hps = list()
        if len(build_hp_combs) > 0:
            hps.extend(list(build_hp_combs[0].keys()))
        if len(other_hp_combs) > 0:
            hps.extend(list(other_hp_combs[0].keys()))
        self._handler = Handler(
            self,
            self._hooks,
            'several_launches',
            evaluation['save_path'],
            evaluation['result_types'],
            eval_dataset_names=[d[1] for d in evaluation['datasets']],
            hyperparameters=hps,
            initial_experiment_counter_value=initial_experiment_counter_value
        )
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
                    # print('(Environment.grid_search)insertion:', insertion)
                    only_build_insertions.append(insertion)
                    shares.append(share)
                build_kwargs = self._pupil_class.form_kwargs(
                    construct(kwargs_for_building),
                    only_build_insertions
                )
                # shared hyperparameters specified as build hps with share field. During build_hp postprocessing share
                # is extracted. Share field applied later

                self._spring_process_for_grid_search(
                    args_for_launches,
                    shares,
                    build_kwargs,
                    session_specs,
                    evaluation,
                    other_hp_combs,
                    build_hp_comb,
                    meta_optimizer_build_kwargs=meta_optimizer_build_kwargs,
                    rebuild_every_time=True
                )
        else:
            self._spring_process_for_grid_search(
                args_for_launches,
                [],
                kwargs_for_building,
                session_specs,
                evaluation,
                other_hp_combs,
                dict(),
                meta_optimizer_build_kwargs=meta_optimizer_build_kwargs,
                rebuild_every_time=True
            )

        self._handler.log_finish_time()
        self._handler.close()

    def _spring_process_for_meta_grid_search(
            self,
            args_for_launches,
            shares,
            pupil_build_kwargs,
            optimizer_build_kwargs,
            session_specs,
            evaluation,
            other_hp_combs,
            base_hp_comb,
            rebuild_every_time=True
    ):
        parsed = configure_args_for_launches(self, args_for_launches, shares, model='meta_optimizer')
        # print("(Environment._spring_process_for_meta_grid_search)parsed:", parsed)
        queue_ = mp.Queue()
        self.mp_debug_flag += 1

        hp_combs = list()
        if len(other_hp_combs) > 0:
            for idx, other_hp_comb in enumerate(other_hp_combs):
                hp_combination = construct(base_hp_comb)
                hp_combination.update(other_hp_comb)
                hp_combs.append(hp_combination)
        else:
            hp_combination = construct(base_hp_comb)
            hp_combs.append(hp_combination)
        order = self._handler.order
        if rebuild_every_time:
            for hp_comb, (start_specs, run_specs_set) in zip(hp_combs, parsed):
                p = mp.Process(
                    target=self._one_optimizer_launch,
                    args=(
                        queue_,
                        pupil_build_kwargs,
                        optimizer_build_kwargs,
                        session_specs,
                        start_specs,
                        run_specs_set,
                        evaluation,
                        hp_comb,
                        order
                    )
                )
                p.start()
                res = queue_.get()
                self._handler.process_results(
                    hp_comb, res, regime='several_meta_optimizer_launches'
                )
                p.join()
        else:
            p = mp.Process(
                target=self._several_optimizer_launches_without_rebuilding,
                args=(
                    queue_,
                    pupil_build_kwargs,
                    optimizer_build_kwargs,
                    session_specs,
                    parsed,
                    evaluation,
                    hp_combs,
                    order
                )
            )
            p.start()
            for hp_comb in hp_combs:
                res = queue_.get()
                self._handler.process_results(
                    hp_comb, res, regime='several_meta_optimizer_launches'
                )
            p.join()

    def grid_search_for_meta(
            self,
            evaluation,
            kwargs_for_pupil_building,
            kwargs_for_optimizer_building,
            build_pupil_hyperparameters=None,
            build_optimizer_hyperparameters=None,
            other_hyperparameters=None,
            initial_experiment_counter_value=0,
            rebuild_every_time=True,
            **kwargs
    ):
        self._store_launch_parameters(
            'optimizer',
            evaluation=evaluation,
            kwargs_for_pupil_building=kwargs_for_pupil_building,
            kwargs_for_optimizer_building=kwargs_for_optimizer_building,
            build_pupil_hyperparameters=build_pupil_hyperparameters,
            build_optimizer_hyperparameters=build_optimizer_hyperparameters,
            other_hyperparameters=other_hyperparameters,
            kwargs=kwargs
        )

        for_evaluation_parsing = construct(evaluation)
        # for batch kwargs filling
        if 'vocabulary' in kwargs:
            for_evaluation_parsing['vocabulary'] = kwargs['vocabulary']
        if 'num_unrollings' in kwargs:
            for_evaluation_parsing['num_unrollings'] = kwargs['num_unrollings']

        # Essential for image batch generators functioning
        if 'valid_batch_kwargs' in kwargs:
            for_evaluation_parsing['valid_batch_kwargs'] = kwargs['valid_batch_kwargs']
        if 'train_batch_kwargs' in kwargs:
            for_evaluation_parsing['train_batch_kwargs'] = kwargs['train_batch_kwargs']
        parsed_evaluation = parse_train_optimizer_method_arguments(
            self, [], for_evaluation_parsing, set_passed_parameters_as_default=False
        )
        evaluation = construct(parsed_evaluation['run'][0]['optimizer_inference'])
        evaluation_save_path = parsed_evaluation['start_specs']['save_path']
        evaluation_result_types = parsed_evaluation['start_specs']['result_types']

        if build_pupil_hyperparameters is None:
            build_pupil_hyperparameters = dict()
        if build_optimizer_hyperparameters is None:
            build_optimizer_hyperparameters = dict()
        if other_hyperparameters is None:
            other_hyperparameters = dict()
        # print("(Environment.grid_search_for_meta)kwargs:", kwargs)
        tmp_output = parse_train_optimizer_method_arguments(
            self,
            [],
            kwargs,
            set_passed_parameters_as_default=False
        )
        session_specs = tmp_output['session_specs']

        build_pupil_hp_combs, build_pupil_insertions = formalize_and_create_insertions_for_build_hps(
            build_pupil_hyperparameters)
        build_optimizer_hp_combs, build_optimizer_insertions = formalize_and_create_insertions_for_build_hps(
            build_optimizer_hyperparameters)
        other_hp_combs, other_insertions = formalize_and_create_insertions_for_other_hps(other_hyperparameters)

        args_for_launches = create_all_args_for_launches(kwargs, other_insertions)
        # print("(Environment.grid_search_for_meta)args_for_launches:", args_for_launches)

        hps = list()
        if len(build_pupil_hp_combs) > 0:
            hps.extend(list(build_pupil_hp_combs[0].keys()))
        if len(build_optimizer_hp_combs) > 0:
            hps.extend(list(build_optimizer_hp_combs[0].keys()))
        if len(other_hp_combs) > 0:
            hps.extend(list(other_hp_combs[0].keys()))
        self._handler = Handler(
            self,
            self._hooks,
            'several_meta_optimizer_launches',
            evaluation_save_path,
            evaluation_result_types,
            eval_pupil_names=nth_element_of_sequence_of_sequences(evaluation['opt_inf_pupil_restore_paths'], 0),
            hyperparameters=hps,
            initial_experiment_counter_value=initial_experiment_counter_value
        )
        self._handler.log_launch()
        if len(build_pupil_hp_combs) > 0:
            for pupil_one_set_of_insertions_and_shares, pupil_build_hp_comb in zip(build_pupil_insertions,
                                                                             build_pupil_hp_combs):
                pupil_build_only_insertions = list()
                pupil_shares = list()
                for insertion, share in pupil_one_set_of_insertions_and_shares:
                    pupil_build_only_insertions.append(insertion)
                    pupil_shares.append(share)
                pupil_build_kwargs = self._pupil_class.form_kwargs(
                    construct(kwargs_for_pupil_building),
                    pupil_build_only_insertions
                )
                if len(build_optimizer_hp_combs) > 0:
                    for optimizer_one_set_of_insertions_and_shares, optimizer_build_hp_comb in zip(
                            build_optimizer_insertions,
                            build_optimizer_hp_combs
                    ):
                        optimizer_build_only_insertions = list()
                        optimizer_shares = list()
                        for insertion, share in optimizer_one_set_of_insertions_and_shares:
                            optimizer_build_only_insertions.append(insertion)
                            optimizer_shares.append(share)
                        optimizer_build_kwargs = self._meta_optimizer_class.form_kwargs(
                            construct(kwargs_for_optimizer_building),
                            optimizer_build_only_insertions
                        )
                        # shared hyperparameters specified as build hps with share field.
                        # During build_hp postprocessing share
                        # is extracted. Share field applied later
                        shares = pupil_shares + optimizer_shares
                        base_hp_comb = construct(pupil_build_hp_comb)
                        base_hp_comb.update(optimizer_build_hp_comb)

                        self._spring_process_for_meta_grid_search(
                            args_for_launches,
                            shares,
                            pupil_build_kwargs,
                            optimizer_build_kwargs,
                            session_specs,
                            evaluation,
                            other_hp_combs,
                            base_hp_comb,
                            rebuild_every_time=rebuild_every_time
                        )
                else:
                    self._spring_process_for_meta_grid_search(
                        args_for_launches,
                        pupil_shares,
                        pupil_build_kwargs,
                        construct(kwargs_for_optimizer_building),
                        session_specs,
                        evaluation,
                        other_hp_combs,
                        construct(pupil_build_hp_comb),
                        rebuild_every_time=rebuild_every_time
                    )
        else:
            if len(build_optimizer_hp_combs) > 0:
                for optimizer_one_set_of_insertions_and_shares, optimizer_build_hp_comb in zip(
                        build_optimizer_insertions,
                        build_optimizer_hp_combs
                ):
                    optimizer_build_only_insertions = list()
                    optimizer_shares = list()
                    for insertion, share in optimizer_one_set_of_insertions_and_shares:
                        optimizer_build_only_insertions.append(insertion)
                        optimizer_shares.append(share)
                    optimizer_build_kwargs = self._meta_optimizer_class.form_kwargs(
                        construct(kwargs_for_optimizer_building),
                        optimizer_build_only_insertions
                    )
                    # shared hyperparameters specified as build hps with share field.
                    # During build_hp postprocessing share
                    # is extracted. Share field applied late

                    self._spring_process_for_meta_grid_search(
                        args_for_launches,
                        optimizer_shares,
                        construct(kwargs_for_pupil_building),
                        optimizer_build_kwargs,
                        session_specs,
                        evaluation,
                        other_hp_combs,
                        optimizer_build_hp_comb,
                        rebuild_every_time=rebuild_every_time
                    )
            else:
                self._spring_process_for_meta_grid_search(
                    args_for_launches,
                    [],
                    construct(kwargs_for_pupil_building),
                    construct(kwargs_for_optimizer_building),
                    session_specs,
                    evaluation,
                    other_hp_combs,
                    dict(),
                    rebuild_every_time=rebuild_every_time
                )
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
                    # print('feed.shape:', feed.shape)
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
        # print('(Environment.one_chat)inq:', inq)
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

    def _optimizer_train_process(
            self,
            q,
            pupil_build,
            optimizer_build,
            launch
    ):
        # print("(Environment._optimizer_train_process)launch:", launch)
        self.build_pupil(**pupil_build)
        self.build_optimizer(**optimizer_build)
        time = self.train_optimizer(**launch)
        q.put(time)

    def _pupil_train_process(
            self,
            q,
            pupil_build,
            optimizer_build,
            launch,
    ):
        self.build_pupil(**pupil_build)
        if optimizer_build is not None:
            self.build_optimizer(**optimizer_build)
        time = self.train(**launch)
        q.put(time)

    def iter_time(
            self,
            steps,
            base,    # time which is used to compute relative effectiveness
            pupil_build_kwargs,
            optimizer_build_kwargs,
            launch_kwargs,
            pupil_varying,
            optimizer_varying,
            launch_varying,
            model='optimizer',
    ):
        # print("(Environment.optimizer_iter_time)optimizer_build_kwargs:", optimizer_build_kwargs)
        result = list()
        launch_kwargs = construct(launch_kwargs)
        launch_kwargs['stop'] = steps
        insertions = form_combinations_from_dicts(
            [pupil_varying, optimizer_varying, launch_varying]
        )
        if model =='optimizer':
            func = self._optimizer_train_process
        else:
            func = self._pupil_train_process
        for insertion_list in insertions:
            queue_ = mp.Queue()

            pupil = construct(pupil_build_kwargs)
            for name, value in insertion_list[0].items():
                pupil[name] = value

            if optimizer_build_kwargs is None:
                optimizer = None
            else:
                optimizer = construct(optimizer_build_kwargs)
                for name, value in insertion_list[1].items():
                    optimizer[name] = value

            launch = construct(launch_kwargs)
            for name, value in insertion_list[2].items():
                launch[name] = value

            p = mp.Process(
                target=func,
                args=(
                    queue_,
                    pupil,
                    optimizer,
                    launch,
                )
            )
            p.start()
            time = queue_.get()
            p.join()
            res = [insertion_list, time / steps]
            if base is not None:
                res.append(res[1] / base)
            result.append(
                res
            )
        return result
