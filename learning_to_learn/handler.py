import os
import time
import datetime as dt

import numpy as np
import tensorflow as tf

from learning_to_learn.useful_functions import create_path, add_index_to_filename_if_needed, construct, nested2string, \
    WrongMethodCallError, extend_dictionary, flatten, check_if_line_is_header, parse_header_line, \
    hyperparameter_name_string, sort_hps, hp_name_2_hp_description, check_if_hp_description_is_in_list
from learning_to_learn.controller import Controller


class Handler(object):

    _stars = '*'*30

    def _compose_prefix(self, prefix):
        res = ''
        # print('(Handler._compose_prefix)self._save_path:', self._save_path)
        if len(self._save_path) > 0:
            res = res + self._save_path + '/'
        if len(prefix) > 0:
            res = res + prefix + '/'
        return res

    def _add_results_file_name_set(self, result_types, prefix='', key_path=None, postfix='train'):
        prefix = self._compose_prefix(prefix)
        d = extend_dictionary(self._file_names, key_path)
        if 'results' not in d:
            d['results'] = dict()
        res = d['results']
        for res_type in result_types:
            file_name = prefix + 'results/%s_' % res_type + postfix + '.txt'
            create_path(file_name, file_name_is_in_path=True)
            res[res_type] = file_name

    def _add_opt_inf_results_file_name_templates(self, prefix='', key_path=None, postfix='train'):
        prefix = self._compose_prefix(prefix)
        d = extend_dictionary(self._file_names, key_path)
        if 'results' not in d:
            d['results'] = dict()
        res = d['results']
        for res_type in self._result_types:
            file_name = prefix + 'results/%s_' % res_type + postfix + '/step%s.txt'
            res[res_type] = file_name
            create_path(file_name, file_name_is_in_path=True)

    def _add_example_file_names(self, prefix='', key_path=None, fuse_file_name=None, example_file_name=None):
        prefix = self._compose_prefix(prefix)
        d = extend_dictionary(self._file_names, key_path)
        if fuse_file_name is not None:
            file_name = prefix + fuse_file_name
            create_path(file_name, file_name_is_in_path=True)
            d['fuses'] = file_name

        if example_file_name is not None:
            file_name = prefix + example_file_name
            create_path(file_name, file_name_is_in_path=True)
            d['examples'] = file_name

    def _get_optimizer_inference_file_name(self, regime, res_type):
        # print("(Handler._get_optimizer_inference_file_name)self._file_names:", self._file_names)
        # print("(Handler._get_optimizer_inference_file_name)self._name_of_pupil_for_optimizer_inference:",
        #       self._name_of_pupil_for_optimizer_inference)
        # print("(Handler._get_optimizer_inference_file_name)regime:", regime)
        # print("(Handler._get_optimizer_inference_file_name)self._meta_optimizer_training_step:",
        #       self._meta_optimizer_training_step)
        return self._file_names[self._name_of_pupil_for_optimizer_inference][regime]['results'][res_type] % \
               self._meta_optimizer_training_step

    def _create_train_fields(self):

        self._controllers = None
        self._results_collect_interval = None
        self._print_per_collected = None
        self._example_per_print = None
        self._train_tensor_schedule = None
        self._validation_tensor_schedule = None

        self._fuses = None
        self._print_fuses = True
        self._fuse_tensor_schedule = None

        self._print_examples = True
        self._example_tensor_schedule = None

        self._processed_fuse_index = None

        self._text_is_being_accumulated = False
        self._accumulated_text = None
        self._accumulated_input = None
        self._accumulated_predictions = None
        self._accumulated_prob_vecs = None

        self._printed_result_types = None
        self._printed_controllers = None

        # print("(Handler._create_train_fields)self._summary:", self._summary)
        # print("(Handler._create_train_fields)self._save_path:", self._save_path)
        # print("(Handler._create_train_fields)self._add_graph_to_summary:", self._add_graph_to_summary)
        if self._summary and self._save_path is not None:
            self._writer = tf.summary.FileWriter(self._save_path + '/' + 'summary')
            if self._add_graph_to_summary:
                self._writer.add_graph(tf.get_default_graph())

        self._training_step = None
        self._accumulation_is_performed = False
        self._accumulated_tensors = dict()
        self._accumulated = dict([(res_key, None) for res_key in self._result_types])

    def __init__(
            self,
            environment_instance,
            hooks,
            processing_type,
            save_path,
            result_types,
            summary=False,
            add_graph_to_summary=False,
            save_to_file=None,
            save_to_storage=None,
            print_results=None,
            batch_generator_class=None,
            vocabulary=None,
            # several_launches method specific
            eval_dataset_names=None,
            eval_pupil_names=None,
            hyperparameters=None,
            initial_experiment_counter_value=0,
            # test method specific
            validation_dataset_names=None,
            validation_tensor_schedule=None,
            printed_result_types=None,
            fuses=None,
            fuse_tensor_schedule=None,
            fuse_file_name=None,
            example_tensor_schedule=None,
            example_file_name=None,
            verbose=True
    ):
        self._verbose = verbose
        if printed_result_types is None:
            printed_result_types = ['loss']
        # print('Initializing Handler! pid = %s' % os.getpid())
        self._processing_type = processing_type
        self._environment_instance = environment_instance
        # print('(Handler.__init__)save_path:', save_path)
        self._save_path = save_path
        self._current_log_path = None
        self._result_types = self._environment_instance.put_result_types_in_correct_order(result_types)
        self._meta_optimizer_train_result_types = None
        self._bpc = 'bpc' in self._result_types
        self._hooks = hooks
        self._last_run_tensor_order = dict()
        self._save_to_file = save_to_file
        self._save_to_storage = save_to_storage
        self._print_results = print_results

        self._batch_generator_class = batch_generator_class
        self._one_char_generation = None

        self._vocabulary = vocabulary
        if self._save_path is not None:
            create_path(self._save_path)

        self._print_order = ['loss', 'bpc', 'perplexity', 'accuracy']

        self._summary = summary
        self._add_graph_to_summary = add_graph_to_summary

        self._file_names = dict()

        self._training_step = None

        self._meta_optimizer_inference_is_performed = False

        if self._processing_type == 'train' or self._processing_type == 'train_with_meta':
            self._create_train_fields()
            if self._save_path is not None:
                self._add_results_file_name_set(self._result_types, key_path=['train'])
                self._add_example_file_names(fuse_file_name=fuse_file_name, example_file_name=example_file_name)
            self._environment_instance.init_storage('train',
                                                    steps=list(),
                                                    loss=list(),
                                                    perplexity=list(),
                                                    accuracy=list(),
                                                    bpc=list())

        if self._processing_type == 'test':
            self._name_of_dataset_on_which_accumulating = None
            self._validation_tensor_schedule = validation_tensor_schedule
            self._validation_dataset_names = validation_dataset_names
            for dataset_name in self._validation_dataset_names:
                self._add_validation_experiment_instruments(dataset_name)
            self._printed_result_types = printed_result_types
            self._accumulated_tensors = dict(
                valid_print_tensors=dict(),
                valid_save_text_tensors=dict()
            )
            # self._accumulated = dict(loss=None, perplexity=None, accuracy=None)
            # if self._bpc:
            #     self._accumulated['bpc'] = None

            self._accumulated = dict([(res_key, None) for res_key in self._result_types])

            self._fuses = fuses
            if self._fuses is not None:
                for fuse in self._fuses:
                    fuse['results'] = list()
            self._fuse_tensor_schedule = fuse_tensor_schedule

            self._processed_fuse_index = None

            self._example_tensor_schedule = example_tensor_schedule

            if self._print_results is None:
                self._print_fuses = False
                self._print_examples = False
            else:
                self._print_fuses = self._print_results
                self._print_examples = self._print_results

            self._text_is_being_accumulated = False
            self._accumulated_text = None
            self._accumulated_input = None
            self._accumulated_predictions = None
            self._accumulated_prob_vecs = None
            if self._save_path is not None:
                self._add_example_file_names(fuse_file_name=fuse_file_name, example_file_name=example_file_name)

        if self._processing_type == 'several_launches':
            self._eval_dataset_names = eval_dataset_names
            self._hyperparameters = hyperparameters
            self._order = list(self._result_types)
            if self._hyperparameters is not None:
                self._order.extend(sort_hps(self._hyperparameters))
            # print("(Handler.__init__)self._order:", self._order)
            self._tmpl = '%s '*(len(self._order) - 1) + '%s\n'
            result_names = list()
            for result_type in self._order:
                if not isinstance(result_type, tuple):
                    result_names.append(result_type)
                else:
                    result_names.append(hyperparameter_name_string(result_type))
            # print("(Handler.__init__)result_names:", result_names)
            # print("(Handler.__init__)self._tmpl:", self._tmpl)
            for dataset_name in eval_dataset_names:
                self._file_names[dataset_name] = os.path.join(self._save_path, dataset_name + '.txt')
                if not os.path.exists(self._file_names[dataset_name]):
                    create_path(self._file_names[dataset_name], file_name_is_in_path=True)
                    with open(self._file_names[dataset_name], 'w') as fd:
                        fd.write(self._tmpl % tuple(result_names))
                elif os.stat(self._file_names[dataset_name]).st_size == 0:
                    with open(self._file_names[dataset_name], 'a') as fd:
                        fd.write(self._tmpl % tuple(result_names))
                else:
                    print("NOT EMPTY EVAL FILE IS PRESENT self._order. MODIFICATION CODE WAS COMMENTED OUT!!!")
                # else:
                #     with open(self._file_names[dataset_name], 'r') as fd:
                #         lines = fd.read().split('\n')
                #     metrics, hp_names = parse_header_line(lines[0])
                #     self._order = metrics + [hp_name_2_hp_description(n) for n in hp_names]

            self._environment_instance.set_in_storage(launches=list())

        if self._processing_type == 'several_meta_optimizer_launches':
            self._eval_pupil_names = eval_pupil_names
            self._hyperparameters = hyperparameters
            self._order = sort_hps(self._hyperparameters)
            result_names = list()
            for result_type in self._order:
                result_names.append(hyperparameter_name_string(result_type))
            with open(self._save_path + '/' + 'hp_layout.txt', 'w') as f:
                layout_str = ''
                for name in result_names[:-1]:
                    layout_str += name + ' '
                layout_str += result_names[-1]
                f.write(layout_str)
            self._experiment_counter = initial_experiment_counter_value
            self._tmpl = '%s/%s/%s_%s.txt'  # <experiment number>/<pupil_name>/<metric>_<regime>.txt
            self._hp_values_str_tmpl = '%s '*(len(self._order) - 1) + '%s'
            self._environment_instance.set_in_storage(launches=list())

        if self._processing_type == 'train_meta_optimizer':
            self._opt_inf_results_collect_interval = None
            self._opt_inf_print_per_collected = None
            self._opt_inf_example_per_print = None

            self._name_of_pupil_for_optimizer_inference = None
            self._meta_optimizer_training_step = None

            self._create_train_fields()
            self._opt_train_results_collect_interval = None
            self._opt_train_print_per_collected = None
            self._opt_train_example_per_print = None

            self._opt_inf_pupil_names = None
            self._meta_optimizer_train_result_types = list()
            print_order_additions = list()
            for res_type in self._print_order:
                if res_type in self._result_types:
                    start_res_type = 'start_' + res_type
                    if start_res_type in self._hooks:
                        if start_res_type not in self._meta_optimizer_train_result_types:
                            self._meta_optimizer_train_result_types.append(start_res_type)
                        if start_res_type not in self._print_order:
                            print_order_additions.append(start_res_type)

                    end_res_type = 'end_' + res_type
                    if end_res_type in self._hooks:
                        if end_res_type not in self._meta_optimizer_train_result_types:
                            self._meta_optimizer_train_result_types.append(end_res_type)
                        if end_res_type not in self._print_order:
                            print_order_additions.append(end_res_type)
            if self._save_path is not None:
                self._add_results_file_name_set(
                    self._meta_optimizer_train_result_types,
                    key_path=['train']
                )
            self._print_order.extend(print_order_additions)

            train_init = dict(
                steps=list()
            )
            for res_type in self._meta_optimizer_train_result_types:
                train_init[res_type] = list()

            self._environment_instance.init_meta_optimizer_training_storage(
                train_init=train_init,
                wipe_storage=False
            )

            self._opt_inf_print_has_already_been_performed = False

        # The order in which tensors are presented in the list returned by get_additional_tensors method
        # It is a list. Each element is either tensor alias or a tuple if corresponding hook is pointing to a list of
        # tensors. Such tuple contains tensor alias, and sizes of nested lists

    @property
    def order(self):
        return construct(self._order)

    def set_pupil_name(self, pupil_name):
        self._name_of_pupil_for_optimizer_inference = pupil_name

    def set_meta_optimizer_training_step(self, step):
        self._meta_optimizer_training_step = step

    def set_meta_optimizer_inference_flags(self, moif, oiphabp):
        self._meta_optimizer_inference_is_performed = moif
        self._opt_inf_print_has_already_been_performed = oiphabp

    def _add_validation_experiment_instruments(self, dataset_name):
        if self._save_path is not None:
            self._add_results_file_name_set(self._result_types, key_path=[dataset_name], postfix=dataset_name)
        init_dict = dict()
        for key in self._result_types + ['steps']:
            init_dict[key] = list()
        # print("(Handler._add_validation_experiment_instruments)dataset_name:", dataset_name)
        if not self._environment_instance.dataset_in_storage(dataset_name):
            self._environment_instance.init_storage(dataset_name, **init_dict)

    def set_new_run_schedule(self, schedule, validation_dataset_names, save_direction='main'):
        # print("(Handler.set_new_run_schedule)validation_dataset_names:", validation_dataset_names)
        self._results_collect_interval = schedule['to_be_collected_while_training']['results_collect_interval']
        if self._results_collect_interval is not None:
            if self._result_types is not None:
                self._save_to_file = True
                self._save_to_storage = True
            else:
                self._save_to_file = False
                self._save_to_storage = False
        else:
            self._save_to_file = False
            self._save_to_storage = False
        self._print_per_collected = schedule['to_be_collected_while_training']['print_per_collected']
        self._example_per_print = schedule['to_be_collected_while_training']['example_per_print']
        if self._example_per_print is not None:
            self._print_fuses = True
            self._print_examples = True
        else:
            self._print_fuses = False
            self._print_examples = False
        self._train_tensor_schedule = schedule['train_tensor_schedule']
        self._validation_tensor_schedule = schedule['validation_tensor_schedule']
        for tensor_use, tensor_instructions in self._validation_tensor_schedule.items():
            self._accumulated_tensors[tensor_use] = dict()
            for tensor_alias, step_schedule in tensor_instructions.items():
                self._accumulated_tensors[tensor_use][tensor_alias] = {'values': list(), 'steps': step_schedule}
        self._printed_result_types = schedule['printed_result_types']

        self._fuses = schedule['fuses']
        self._fuse_tensor_schedule = schedule['fuse_tensors']
        if self._fuses is not None:
            for fuse in self._fuses:
                fuse['results'] = list()

        self._example_tensor_schedule = schedule['example_tensors']

        if self._save_path is not None:
            if save_direction == 'main':
                self._add_example_file_names()
            else:
                self._add_example_file_names(prefix=save_direction, key_path=[save_direction])

        if self._printed_result_types is not None:
            if len(self._printed_result_types) > 0:
                self._print_results = True
        self._printed_controllers = schedule['printed_controllers']
        for dataset_name in validation_dataset_names:
            self._add_validation_experiment_instruments(dataset_name)

    def set_optimizer_train_schedule(
            self, schedule,
            opt_inf_pupil_names=None,
            opt_inf_to_be_collected_while_training=None,
            opt_inf_train_tensor_schedule=None,
            opt_inf_validation_tensor_schedule=None
    ):
        # print("(Handler.set_optimizer_train_schedule)self._opt_inf_pupil_names:", self._opt_inf_pupil_names)
        self._opt_inf_pupil_names = opt_inf_pupil_names
        if self._save_path is not None:
            if self._opt_inf_pupil_names is not None:
                for pupil_name in self._opt_inf_pupil_names:
                    if self._save_path is not None:
                        self._add_opt_inf_results_file_name_templates(
                            prefix=pupil_name, key_path=[pupil_name, 'train'], postfix='train')
                        self._add_opt_inf_results_file_name_templates(
                            prefix=pupil_name, key_path=[pupil_name, 'validation'], postfix='validation')

        if schedule is not None:
            self._opt_train_results_collect_interval = \
                schedule['to_be_collected_while_training']['results_collect_interval']
            # print("(Handler.set_optimizer_train_schedule)self._results_collect_interval:", self._results_collect_interval)
            if self._opt_train_results_collect_interval is not None:
                if self._result_types is not None:
                    self._save_to_file = True
                    self._save_to_storage = True
                else:
                    self._save_to_file = False
                    self._save_to_storage = False
            else:
                self._save_to_file = False
                self._save_to_storage = False
            self._opt_train_print_per_collected = schedule['to_be_collected_while_training']['print_per_collected']

            # print("(Handler.set_optimizer_train_schedule)schedule['train_tensor_schedule']:",
            #       schedule['train_tensor_schedule'])
            self._train_tensor_schedule = schedule['train_tensor_schedule']

            self._printed_result_types = schedule['printed_result_types']

            self._printed_controllers = schedule['printed_controllers']

        if self._printed_result_types is not None:
            if len(self._printed_result_types) > 0:
                self._print_results = True

        meta_optimizer_print_result_types = list()
        for res_type in self._printed_result_types:
            start_res_type = 'start_' + res_type
            end_res_type = 'end_' + res_type
            meta_optimizer_print_result_types.append(start_res_type)
            meta_optimizer_print_result_types.append(end_res_type)
        self._printed_result_types.extend(meta_optimizer_print_result_types)

        self._environment_instance.init_meta_optimizer_training_storage(
            self._opt_inf_pupil_names,
            # opt_inf_init=self._environment_instance.create_opt_inf_init(self._result_types)
        )
        if opt_inf_to_be_collected_while_training is not None:
            self._opt_inf_results_collect_interval = opt_inf_to_be_collected_while_training[
                'opt_inf_results_collect_interval']
            self._opt_inf_print_per_collected = opt_inf_to_be_collected_while_training['opt_inf_print_per_collected']
            self._opt_inf_example_per_print = opt_inf_to_be_collected_while_training['opt_inf_example_per_print']

    def set_controllers(self, controllers):
        # print("(Handler.set_controllers)controllers:", controllers)
        self._controllers = controllers

    def start_accumulation(self, dataset_name, training_step=None):
        self._name_of_dataset_on_which_accumulating = dataset_name
        self._training_step = training_step
        for res_type in self._accumulated.keys():
            self._accumulated[res_type] = list()

    @staticmethod
    def decide(higher_bool, lower_bool):
        if higher_bool is None:
            if lower_bool is None:
                answer = False
            else:
                answer = lower_bool
        else:
            answer = higher_bool
        return answer

    def stop_accumulation(
            self,
            save_to_file=True,
            save_to_storage=True,
            print_results=True,
            file_open_mode='a',
    ):
        save_to_file = self.decide(save_to_file, self._save_to_file)
        save_to_storage = self.decide(save_to_storage, self._save_to_storage)
        print_results = self.decide(print_results, self._print_results)
        means = dict()
        for key, value_list in self._accumulated.items():
            #print('%s:' % key, value_list)
            counter = 0
            mean = 0
            for value in value_list:
                if isinstance(value, tuple):
                    if value[0] >= 0.:
                        mean += value[0] * value[1]
                        counter += value[1]
                else:
                    if value >= 0.:
                        mean += value
                        counter += 1
            if counter == 0:
                mean = 0.
            else:
                mean = mean / counter
            # print('(stop_accumulation)counter:', counter)
            if self._save_path is not None:
                if save_to_file:
                    if self._meta_optimizer_inference_is_performed:
                        file_name = self._get_optimizer_inference_file_name('validation', key)
                    else:
                        file_name = self._file_names[self._name_of_dataset_on_which_accumulating]['results'][key]
                    with open(file_name, file_open_mode) as f:
                        if self._training_step is not None:
                            f.write('%s %s\n' % (self._training_step, mean))
                        else:
                            try:
                                f.write('%s\n' % mean)
                            except TypeError:
                                print('(Handler.stop_accumulation)value_list:', value_list)
                                raise
            means[key] = mean
        if save_to_storage:
            # if self._meta_optimizer_inference_is_performed:
            #     self._environment_instance.append_to_optimizer_inference_storage(
            #         self._name_of_pupil_for_optimizer_inference, 'validation', self._meta_optimizer_training_step,
            #         **dict([(key, means[key]) for key in self._result_types])
            #     )
            # else:
            for_storage = construct(means)
            for_storage['steps'] = self._training_step
            self._environment_instance.append_to_storage(
                self._name_of_dataset_on_which_accumulating,
                **dict([(key, for_storage[key]) for key in self._result_types + ['steps']])
            )
        if print_results:
            self._print_standard_report(
                regime='validation',
                message='results on validation dataset %s' % self._name_of_dataset_on_which_accumulating,
                **means)
        if 'valid_print_tensors' in self._accumulated_tensors:
            valid_print_tensors = self._accumulated_tensors['valid_print_tensors']
            if len(valid_print_tensors) > 0:
                self._print_validation_tensors(valid_print_tensors)
        self._training_step = None
        self._name_of_dataset_on_which_accumulating = None
        self._save_accumulated_tensors()
        return means

    def set_processed_fuse_index(self, fuse_idx):
        self._processed_fuse_index = fuse_idx

    def start_fuse_accumulation(self):
        self._accumulated_text = ''
        self._text_is_being_accumulated = True

    def stop_fuse_accumulation(self):
        # print('self._fuses:', self._fuses)
        self._fuses[self._processed_fuse_index]['results'].append(str(self._accumulated_text))
        self._accumulated_text = None
        self._text_is_being_accumulated = False

    def start_example_accumulation(self):
        self._text_is_being_accumulated = True
        if self._batch_generator_class.__name__ == 'BpeBatchGenerator' or \
                        self._batch_generator_class.__name__ == 'BpeBatchGeneratorOneHot' or \
                        self._batch_generator_class.__name__ == 'NgramsBatchGenerator' or \
                        self._batch_generator_class.__name__ == 'BpeFastBatchGenerator' or \
                        self._batch_generator_class.__name__ == 'BpeFastBatchGeneratorOneHot' or \
                        self._batch_generator_class.__name__ == 'NgramsFastBatchGenerator':
            self._one_char_generation = False
        else:
            self._one_char_generation = True
        self._accumulated_prob_vecs = list()
        if self._one_char_generation:
            self._accumulated_input = ''
            self._accumulated_predictions = ''
        else:
            self._accumulated_input = list()
            self._accumulated_predictions = list()

    def stop_example_accumulation(self):
        # print('self._fuses:', self._fuses)
        self._text_is_being_accumulated = False

    def _prepare_string(self, res):
        preprocessed = ''
        for char in res:
            preprocessed += self._form_string_char(char)
        return preprocessed

    def _form_fuse_msg(self, training_step):
        msg = ''
        if training_step is not None:
            msg += 'generation from fuses on step %s' % str(training_step) + '\n'
        else:
            msg += 'generation from fuses\n'
        msg += (self._stars + '\n') * 2
        for fuse in self._fuses:
            msg += ('\nfuse: ' + fuse['text'] + '\n')
            msg += self._stars + '\n'
            for res_idx, res in enumerate(fuse['results']):
                if res_idx > 0:
                    msg += ('\nlaunch number: ' + str(res_idx) + '\n')
                else:
                    msg += ('launch number: ' + str(res_idx) + '\n')
                msg += ('result: ' + self._prepare_string(res) + '\n')
            msg += self._stars + '\n'*2
        msg += (self._stars + '\n') * 2
        return msg

    def _form_example_msg(self, training_step):
        # print('inside Handler._form_example_msg')
        # print('(Handler._form_example_msg)self._accumulated_input:', self._accumulated_input)
        # print('(Handler._form_example_msg)self._accumulated_predictions:', self._accumulated_predictions)
        msg = ''
        if training_step is not None:
            msg += 'example generation on step %s' % str(training_step) + '\n'
        else:
            msg += 'example generation\n'
        msg += (self._stars + '\n')
        if self._one_char_generation:
            msg += 'input:\n'
            msg += (self._accumulated_input + '\n')
            msg += 'predictions:\n'
            msg += (self._accumulated_predictions + '\n')
        else:
            for idx, (inp, pred) in enumerate(zip(self._accumulated_input, self._accumulated_predictions)):
                msg += '%s.|%s|%s|\n' % (idx, inp, pred)
        msg += self._stars + '\n'
        return msg

    def _print_fuse_results(self, training_step):
        print(self._form_fuse_msg(training_step))

    def _save_fuse_results(self, training_step):
        with open(self._file_names['fuses'], 'a', encoding='utf-8') as f:
            f.write(self._form_fuse_msg(training_step) + '\n'*2)

    def _print_example_results(self, training_step):
        print(self._form_example_msg(training_step))

    def _save_example_results(self, training_step):
        with open(self._file_names['examples'], 'a', encoding='utf-8') as f:
            f.write(self._form_example_msg(training_step) + '\n'*2)

    def clean_fuse_results(self):
        for fuse in self._fuses:
            fuse['results'] = list()

    def dispense_fuse_results(self, training_step):
        if self._print_fuses:
            self._print_fuse_results(training_step)
        if self._save_path is not None:
            self._save_fuse_results(training_step)
        res = construct(self._fuses)
        self.clean_fuse_results()
        return res

    def dispense_example_results(self, training_step):
        # print('(Handler.dispense_example_results)self._print_examples:', self._print_examples)
        if self._print_examples:
            self._print_example_results(training_step)
        if self._save_path is not None:
            self._save_example_results(training_step)
        if self._one_char_generation:
            acc_inp = str(self._accumulated_input)
            acc_out = str(self._accumulated_predictions)
        else:
            acc_inp = ''.join(self._accumulated_input)
            acc_out = ''.join(self._accumulated_predictions)
        res = {'input': acc_inp,
               'output': acc_out,
               'prob_vecs': construct(self._accumulated_prob_vecs)}
        self._accumulated_input = None
        self._accumulated_predictions = None
        return res

    @staticmethod
    def _print_1_validation_hook_result(hook_res):
        if isinstance(hook_res, list):
            for high_idx, high_elem in enumerate(hook_res):
                if isinstance(high_elem, list):
                    for low_idx, low_elem in enumerate(high_elem):
                        print('\n'*4 + '[%s][%s]:' % (high_idx, low_idx), low_elem)
                else:
                    print('\n'*2 + '[%s]:' % high_idx, high_elem)
        else:
            print(hook_res)

    def _print_validation_tensors(self, valid_print_tensors):
        print('validation tensors:')
        for tensor_alias, res in valid_print_tensors.items():
            print(tensor_alias + ':')
            if isinstance(res['steps'], int):
                steps = [res['steps'] * i for i in range(len(res['values']))]
            if isinstance(res['steps'], list):
                steps = res['steps']
            for step, value in zip(steps, res['values']):
                print('%s:' % step)
                self._print_1_validation_hook_result(value)
            print('')

    def _process_validation_results(self,
                                    step,
                                    validation_res):
        # print("self._last_run_tensor_order['basic']['borders']:", self._last_run_tensor_order['basic']['borders'])
        tmp_output = validation_res[self._last_run_tensor_order['basic']['borders'][0] + 1:
            self._last_run_tensor_order['basic']['borders'][1]]
        # print('tmp_output:', tmp_output)
        self._accumulate_several_data(self._result_types, tmp_output)
        # if self._bpc:
        #     [loss, perplexity, accuracy, bpc] = tmp_output
        #     self._accumulate_several_data(['loss', 'perplexity', 'accuracy', 'bpc'], [loss, perplexity, accuracy, bpc])
        # else:
        #     [loss, perplexity, accuracy] = tmp_output
        #     self._accumulate_several_data(['loss', 'perplexity', 'accuracy'], [loss, perplexity, accuracy])
        self._accumulate_tensors(step, validation_res)

    @staticmethod
    def _comp_chr_acc_of_2_tokens(correct_token, output_token):
        length = max(len(correct_token), len(output_token))
        corr_chrs_num = 0
        for idx in range(min(len(correct_token), len(output_token))):
            if correct_token[idx] == output_token[idx]:
                corr_chrs_num += 1
        # print('(Handler._comp_chr_acc_of_2_tokens)return:', corr_chrs_num // length)
        # print('(Handler._comp_chr_acc_of_2_tokens)correct and output tokens, accuracy:',
        #       (correct_token, output_token, corr_chrs_num / length))
        return corr_chrs_num / length

    def _process_validation_by_chars_results(
            self, step, validation_res, correct_token):
        correct_token = ''.join(correct_token)
        # print('(Handler._process_validation_by_chars_results)entered processing')
        tmp_output = validation_res[self._last_run_tensor_order['basic']['borders'][0]:
            self._last_run_tensor_order['basic']['borders'][1]]
        if self._bpc:
            [prediction, loss, perplexity, _, bpc] = tmp_output
            output_token = self._batch_generator_class.vec2char(prediction, self._vocabulary)[0]
            self._accumulate_several_data(
                ['loss', 'perplexity', 'accuracy', 'bpc'],
                [loss, perplexity,
                 (self._comp_chr_acc_of_2_tokens(correct_token, output_token), len(correct_token)),
                 bpc])
        else:
            [prediction, loss, perplexity, _] = tmp_output
            output_token = self._batch_generator_class.vec2char(prediction, self._vocabulary)[0]
            self._accumulate_several_data(
                ['loss', 'perplexity', 'accuracy'],
                [loss, perplexity,
                 (self._comp_chr_acc_of_2_tokens(correct_token, output_token), len(correct_token))])
        self._accumulate_tensors(step, validation_res)

    def _cope_with_tensor_alias(self,
                                alias):
        # print("(Handler._cope_with_tensor_alias)alias:", alias)
        # print("(Handler._cope_with_tensor_alias)self._hooks[alias]:", self._hooks[alias])
        if not isinstance(self._hooks[alias], list):
            return [self._hooks[alias]], 1
        add_tensors = list()
        order = [alias, len(self._hooks[alias])]
        counter = 0
        if len(self._hooks[alias]) > 0:
            if isinstance(self._hooks[alias][0], list):
                order.append(len(self._hooks[alias][0]))
                for elem in self._hooks[alias]:
                    for tensor in elem:
                        add_tensors.append(tensor)
                        counter += 1
            for tensor in self._hooks[alias]:
                add_tensors.append(tensor)
                counter += 1
        # print("(Handler._cope_with_tensor_alias)add_tensors:", add_tensors)
        # print("(Handler._cope_with_tensor_alias)counter:", counter)
        return add_tensors, counter

    def _save_datum(self, descriptor, step, datum, processing_type, dataset_name):
        # print("(Handler._save_datum)self._file_names:", self._file_names)
        # print("(Handler._save_datum)cwd:", os.getcwd())
        if self._meta_optimizer_inference_is_performed:
            file_name = self._get_optimizer_inference_file_name(processing_type, descriptor)
        else:
            if processing_type == 'train':
                file_name = self._file_names['train']['results'][descriptor]
            elif processing_type == 'validation':
                file_name = self._file_names[dataset_name]['results'][descriptor]

        with open(file_name, 'a') as f:
            f.write('%s %s\n' % (step, datum))

    def _save_launch_results(self, results, hp):
        for dataset_name, res in results.items():
            values = list()
            all_together = dict(hp)
            # print('dataset_name:', dataset_name)
            # print("(Handler._save_launch_results)all_together:", all_together)
            # print("(Handler._save_launch_results)self._order:", self._order)
            # print("(Handler._save_launch_results)list(hp.keys()):", list(hp.keys()))
            all_together.update(res)
            for key in self._order:
                if isinstance(key, tuple):
                    # print("(Handler._save_launch_results)key:", key)
                    present, matched_key = check_if_hp_description_is_in_list(key, list(hp.keys()))
                else:
                    present, matched_key = True, key
                if present:
                    values.append(all_together[matched_key])
            with open(self._file_names[dataset_name], 'a') as f:
                f.write(self._tmpl % tuple(values))

    def _save_optimizer_launch_results(self, results, hp):
        if self._save_path is not None:
            if len(self._save_path) == 0:
                prefix = ''
            else:
                prefix = self._save_path + '/'
            hp = dict(hp)
            hp_file_name = prefix + str(self._experiment_counter) + '.txt'
            with open(hp_file_name, 'w') as f:
                hp_values = list()
                hp_types = list()
                for key in self._order:
                    hp_values.append(hp[key])
                    hp_types.append(hp[key].__class__.__name__)
                f.write(self._hp_values_str_tmpl % tuple(hp_values))
                f.write('\n')
                f.write(self._hp_values_str_tmpl % tuple(hp_types))
            # print("(Handler._save_optimizer_launch_results)results:", results)
            for pupil_name, pupil_res in results.items():
                for regime, regime_res in pupil_res.items():
                    if regime != 'step':
                        res_types = list(regime_res.keys())
                        for res_type in res_types:
                            if res_type != 'steps':
                                file_name = prefix + \
                                            self._tmpl % (self._experiment_counter, pupil_name, res_type, regime)
                                create_path(file_name, file_name_is_in_path=True)
                                with open(file_name, 'w') as f:
                                    for step, value in zip(regime_res['steps'], regime_res[res_type]):
                                        f.write('%s %s\n' % (step, value))
        self._experiment_counter += 1

    def _save_several_data(self,
                           descriptors,
                           step,
                           data,
                           processing_type='train',
                           dataset_name=None):
        # print("(Handler._save_several_data)descriptors:", descriptors)
        for descriptor, datum in zip(descriptors, data):
            if datum is not None:
                self._save_datum(descriptor, step, datum, processing_type, dataset_name)

    def _save_accumulated_tensors(self):
        pass

    def _accumulate_several_data(self, descriptors, data):
        for descriptor, datum in zip(descriptors, data):
            if datum is not None:
                self._accumulated[descriptor].append(datum)

    def get_tensors(self, regime, step, with_meta_optimizer=False):
        tensors = list()
        self._last_run_tensor_order = dict()
        pointer = 0
        current = dict()
        self._last_run_tensor_order['basic'] = current
        current['tensors'] = dict()
        start = pointer
        if regime == 'train':
            if with_meta_optimizer:
                tensors.append(self._hooks['train_with_meta_optimizer_op'])
                current['tensors']['train_with_meta_optimizer_op'] = [pointer, pointer+1]
                pointer += 1
            else:
                tensors.append(self._hooks['train_op'])
                current['tensors']['train_op'] = [pointer, pointer + 1]
                pointer += 1
            for res_type in self._result_types:
                tensors.append(self._hooks[res_type])
                current['tensors'][res_type] = [pointer, pointer + 1]
                pointer += 1
            self._last_run_tensor_order['basic']['borders'] = [start, pointer]

            if self._train_tensor_schedule is not None:
                additional_tensors = self._get_additional_tensors(self._train_tensor_schedule, step, pointer)
                tensors.extend(additional_tensors)
        if regime == 'validation':
            tensors.append(self._hooks['validation_predictions'])
            current['tensors']['validation_predictions'] = [pointer, pointer + 1]
            pointer += 1
            for res_type in self._result_types:
                tensors.append(self._hooks['validation_' + res_type])
                current['tensors']['validation_' + res_type] = [pointer, pointer + 1]
                pointer += 1
            self._last_run_tensor_order['basic']['borders'] = [start, pointer]

            if self._validation_tensor_schedule is not None:
                additional_tensors = self._get_additional_tensors(self._validation_tensor_schedule, step, pointer)
                tensors.extend(additional_tensors)
        if regime == 'fuse':
            tensors.append(self._hooks['validation_predictions'])
            current['tensors']['validation_predictions'] = [pointer, pointer + 1]
            pointer += 1
            self._last_run_tensor_order['basic']['borders'] = [start, pointer]
            if self._fuse_tensor_schedule is not None:
                additional_tensors = self._get_additional_tensors(self._fuse_tensor_schedule, step, pointer)
                tensors.extend(additional_tensors)
        if regime == 'example':
            tensors.append(self._hooks['validation_predictions'])
            current['tensors']['validation_predictions'] = [pointer, pointer + 1]
            pointer += 1
            self._last_run_tensor_order['basic']['borders'] = [start, pointer]
            if self._example_tensor_schedule is not None:
                additional_tensors = self._get_additional_tensors(self._example_tensor_schedule, step, pointer)
                tensors.extend(additional_tensors)
        if regime == 'train_meta_optimizer':
            tensors.append(self._hooks['optimizer_train_op'])
            current['tensors']['optimizer_train_op'] = [pointer, pointer + 1]
            pointer += 1
            for res_type in self._meta_optimizer_train_result_types:
                tensors.append(self._hooks[res_type])
                current['tensors'][res_type] = [pointer, pointer + 1]
                pointer += 1
            self._last_run_tensor_order['basic']['borders'] = [start, pointer]
            if self._train_tensor_schedule is not None:
                # print("(Handler.get_tensors)self._train_tensor_schedule:", self._train_tensor_schedule)
                additional_tensors = self._get_additional_tensors(self._train_tensor_schedule, step, pointer)
                tensors.extend(additional_tensors)
        # print(tensors)
        return tensors

    def _get_additional_tensors(self,
                                schedule,
                                step,
                                start_pointer):
        # print('_get_additional_tensors method. schedule:', schedule)
        additional_tensors = list()
        pointer = start_pointer
        for tensors_use, tensors_schedule in schedule.items():
            # print('(Handler._get_additional_tensors)tensors_use:', tensors_use)
            # print('(Handler._get_additional_tensors)tensor_schedule:', tensors_schedule)
            self._last_run_tensor_order[tensors_use] = dict()
            self._last_run_tensor_order[tensors_use]['tensors'] = dict()
            start = pointer
            if isinstance(tensors_schedule, dict):
                for tensor_alias, tensor_schedule in tensors_schedule.items():
                    if isinstance(tensor_schedule, list):
                        if step in tensor_schedule:
                            add_tensors, counter = self._cope_with_tensor_alias(tensor_alias)
                            additional_tensors.extend(add_tensors)
                            self._last_run_tensor_order[tensors_use]['tensors'][tensor_alias] = [pointer,
                                                                                                 pointer + counter]
                            pointer += counter
                    elif isinstance(tensor_schedule, int):
                        if step % tensor_schedule == 0:
                            add_tensors, counter = self._cope_with_tensor_alias(tensor_alias)
                            additional_tensors.extend(add_tensors)
                            self._last_run_tensor_order[tensors_use]['tensors'][tensor_alias] = [pointer,
                                                                                                 pointer + counter]
                            pointer += counter
            elif isinstance(tensors_schedule, list):
                for tensor_alias in tensors_schedule:
                    add_tensors, counter = self._cope_with_tensor_alias(tensor_alias)
                    additional_tensors.extend(add_tensors)
                    self._last_run_tensor_order[tensors_use]['tensors'][tensor_alias] = [pointer,
                                                                                         pointer + counter]
                    pointer += counter
            self._last_run_tensor_order[tensors_use]['borders'] = [start, pointer]
        return additional_tensors

    @staticmethod
    def _print_tensors(instructions, print_step_number=False, indent=0):
        if print_step_number:
            print('\n'*indent + 'step:', instructions['step'])
        print(instructions['message'])
        for alias, res in sorted(instructions['results'].items(), key=lambda item: item[0]):
            if not isinstance(res, list):
                print('%s:' % alias, res)
            else:
                print('%s:' % alias)
                if isinstance(res[0], list):
                    for high_idx, high_elem in res.items():
                        print('\n\n[%s]:' % high_idx)
                        for low_idx, low_elem in high_elem.items():
                            print('\n'*4 + '[%s][%s]:' % (high_idx, low_idx), low_elem)
                else:
                    for idx, elem in res.items():
                        print('\n\n[%s]:' % idx, elem)

    @staticmethod
    def print_hyper_parameters(hp, order, indent=2):
        # print("(Handler.print_hyper_parameters)hp:", hp)
        if indent != 0:
            print('\n' * (indent - 1))
        # print("(Handler.print_hyper_parameters)order:", order)
        if len(hp) == 0:
            print("(Handler.print_hyper_parameters)No hyper parameters!")
        for key in order:
            if any([list(key[1:]) == list(k[1:]) for k in hp.keys()]):
                for k in hp.keys():
                    if list(key[1:]) == list(k[1:]):
                        matching_key = k
                print('%s: %s' % (hyperparameter_name_string(matching_key), hp[matching_key]))

    def _print_launch_results(self, results, hp, idx=None, indent=2):
        # print("(Handler._print_launch_results)hp:", hp)
        self.print_hyper_parameters(hp, self._order, indent=indent)
        for dataset_name, res in results.items():
            print('results on %s dataset:' % dataset_name)
            for key in self._order:
                if key in res:
                    print('%s: %s' % (key, res[key]))

    def _accumulate_tensors(self, step, tensors):
        # print('(Handler._accumulate_tensors)self._last_run_tensor_order:', self._last_run_tensor_order)
        tensor_order = construct(self._last_run_tensor_order)
        del tensor_order['basic']
        for tensor_use, instructions_1_use in tensor_order.items():
            current = self._accumulated_tensors[tensor_use]
            extracted = self._extract_results(tensor_order, tensor_use, tensors)
            for tensor_alias, value in extracted.items():
                current[tensor_alias]['values'].append(value)

    def _save_tensors(self, tensors):
        pass

    def _print_controllers(self):
        if self._controllers is not None:
            for controller in self._controllers:
                # if isinstance(controller, Controller):
                #     print("(Handler._print_controllers)controller.name:", controller.name)
                # if isinstance(controller, list):
                #     for c in controller:
                #         if isinstance(c, Controller):
                #             print("(Handler._print_controllers)c.name:", c.name)
                #         else:
                #             print("(Handler._print_controllers)c:", c)
                if controller.name in self._printed_controllers:
                    print('%s:' % controller.name, controller.get())

    def _print_standard_report(self,
                               indents=None,
                               regime='train',
                               **kwargs):
        if indents is None:
            indents = [0, 0]
        for _ in range(indents[0]):
            print()
        if 'time_elapsed' in kwargs:
            print("time elapsed:", kwargs['time_elapsed'])
        print("current time in seconds: %s; date: %s" % (time.clock(), str(dt.datetime.now())))
        if regime == 'train':
            if 'step' in kwargs:
                print('step:', kwargs['step'])
            self._print_controllers()
        if 'message' in kwargs:
            print(kwargs['message'])
        for key, value in kwargs.items():
            if key != 'tensors' \
                    and key != 'step' \
                    and key != 'message' \
                    and key != 'time_elapsed' \
                    and key in self._printed_result_types \
                    and key not in self._print_order:
                print('%s:' % key, value)
        for key in self._print_order:
            if key in kwargs:
                print('%s:' % key, kwargs[key])
        if 'tensors' in kwargs:
            self._print_tensors(kwargs['tensors'], self._train_tensor_schedule)
        for _ in range(indents[1]):
            print('')

    def _get_structure_of_hook(self, alias):
        if not isinstance(self._hooks[alias], list):
            return 1
        else:
            if not isinstance(self._hooks[alias][0], list):
                return [len(self._hooks[alias])]
            else:
                output = [len(self._hooks[alias])]
                for l in self._hooks[alias]:
                    output.append(len(l))
                return output

    def _extract_results(self, last_order, tensor_use, res):
        extracted = dict()
        for alias, borders in last_order[tensor_use]['tensors'].items():
            structure = self._get_structure_of_hook(alias)
            if isinstance(structure, int):
                extracted[alias] = res[borders[0]]
            elif isinstance(structure, list):
                if len(structure) == 1:
                    extracted[alias] = res[borders[0]:borders[1]]
                else:
                    structured = list()
                    pointer = borders[0]
                    for length in structure[1:]:
                        structured.append(res[pointer:pointer+length])
                    extracted[alias] = structured
        return extracted

    def _add_summaries(self, extracted, step):
        # print('\n(Handler._add_summaries)step:', step)
        for ok, ov in extracted.items():
            for v in flatten(ov):
                # print('\n(Handler._add_summaries)%s:' % ok, flatten(ov))
                self._writer.add_summary(v, step)

    def _form_train_tensor_print_instructions(self, step, train_res, last_order):
        instructions = dict()
        instructions['step'] = step
        instructions['message'] = 'train tensors:'
        instructions['results'] = dict()
        extracted_for_printing = self._extract_results(last_order, 'train_print_tensors', train_res)
        # print('extracted_for_printing:', extracted_for_printing)
        instructions['results'].update(extracted_for_printing)
        return instructions

    @staticmethod
    def _toss_train_results(
            res,
            result_types
    ):
        d = dict()
        for r, res_type in zip(res, result_types):
            d[res_type] = r
        return d

    def _process_train_results(self,
                               step,
                               train_res,
                               result_types,
                               results_collect_interval,
                               print_per_collected,
                               time_elapsed,
                               msg='results on train dataset'):
        # print('step:', step)
        # print('train_res:', train_res)
        # print("(Handler._process_train_results)self._last_run_tensor_order:", self._last_run_tensor_order)
        basic_borders = self._last_run_tensor_order['basic']['borders']
        tmp = train_res[basic_borders[0]+1:basic_borders[1]]
        # print("(Handler._process_train_results)result_types:", result_types)
        res_dict = self._toss_train_results(tmp, result_types)
        to_print = res_dict.copy()
        to_print['time_elapsed'] = time_elapsed
        if self._printed_result_types is not None:
            if results_collect_interval is not None:
                if step % (results_collect_interval * print_per_collected) == 0:
                    if self._meta_optimizer_inference_is_performed and \
                            not self._opt_inf_print_has_already_been_performed:
                        indents = [0, 0]
                        self._opt_inf_print_has_already_been_performed = True
                    else:
                        indents = [2, 0]
                    # print("(Handler._process_train_results)res_dict:", res_dict)
                    self._print_standard_report(
                        indents=indents,
                        step=step,
                        message=msg,
                        **to_print
                    )
        if results_collect_interval is not None:
            if step % results_collect_interval == 0:
                if self._save_path is not None:
                    self._save_several_data(result_types,
                                            step,
                                            tmp)
                # if self._meta_optimizer_inference_is_performed:
                #     # print("(Handler._process_train_results)res_dict:", res_dict)
                #     self._environment_instance.append_to_optimizer_inference_storage(
                #         self._name_of_pupil_for_optimizer_inference, 'train', self._meta_optimizer_training_step,
                #         steps=step,
                #         **res_dict
                #     )
                # else:
                self._environment_instance.append_to_storage('train',
                                                             steps=step,
                                                             **res_dict)

        # print("(Handler._process_train_results)self._last_run_tensor_order:", self._last_run_tensor_order)
        if 'train_print_tensors' in self._last_run_tensor_order:
            print_borders = self._last_run_tensor_order['train_print_tensors']['borders']
            if print_borders[1] - print_borders[0] > 0:
                print_instructions = self._form_train_tensor_print_instructions(step,
                                                                                train_res,
                                                                                self._last_run_tensor_order)
                other_stuff_is_printed = (step % (results_collect_interval * print_per_collected) == 0)
                if other_stuff_is_printed:
                    indent = 0
                else:
                    indent = 1
                self._print_tensors(print_instructions,
                                    print_step_number=not other_stuff_is_printed,
                                    indent=indent)
        if 'train_summary_tensors' in self._last_run_tensor_order:
            summary_borders = self._last_run_tensor_order['train_summary_tensors']['borders']
            # print("(Handler._process_train_results)summary_borders:", summary_borders)
            if summary_borders[1] - summary_borders[0] > 0:
                extracted_for_summary = self._extract_results(
                    self._last_run_tensor_order,
                    'train_summary_tensors',
                    train_res
                )
                self._add_summaries(extracted_for_summary, step)

    @staticmethod
    def _form_string_char(char):
        special_characters_map = {'\n': '\\n',
                                  '\t': '\\t'}
        if char in list(special_characters_map.keys()):
            return special_characters_map[char]
        else:
            return char

    def _form_fuse_tensor_print_instructions(self, step, char, fuse_res, last_order):
        instructions = dict()
        instructions['step'] = step
        instructions['message'] = 'fuse tensors:\nchar = %s' % self._form_string_char(char)
        instructions['results'] = dict()
        extracted_for_printing = self._extract_results(last_order, 'fuse_print_tensors', fuse_res)
        # print('extracted_for_printing:', extracted_for_printing)
        instructions['results'].update(extracted_for_printing)
        return instructions

    def _form_example_tensor_print_instructions(self, step, char, example_res, last_order):
        instructions = dict()
        instructions['step'] = step
        instructions['message'] = 'example tensors:\nchar = %s' % self._form_string_char(char)
        instructions['results'] = dict()
        extracted_for_printing = self._extract_results(last_order, 'example_print_tensors', example_res)
        # print('extracted_for_printing:', extracted_for_printing)
        instructions['results'].update(extracted_for_printing)
        return instructions

    def _process_fuse_generation_results(self, step, res):
        basic_borders = self._last_run_tensor_order['basic']['borders']
        [prediction] = res[basic_borders[0]:basic_borders[1]]
        char = self._batch_generator_class.vec2char(prediction, self._vocabulary)[0]
        if self._text_is_being_accumulated:
            self._accumulated_text += char
        if 'fuse_print_tensors' in self._last_run_tensor_order:
            print_borders = self._last_run_tensor_order['fuse_print_tensors']['borders']
            if print_borders[1] - print_borders[0] > 0:
                print_instructions = self._form_fuse_tensor_print_instructions(step,
                                                                               char,
                                                                               res,
                                                                               self._last_run_tensor_order)
                self._print_tensors(print_instructions,
                                    print_step_number=True,
                                    indent=1)

    def _process_example_generation_results(self, step, input_str, res):
        basic_borders = self._last_run_tensor_order['basic']['borders']
        [prediction] = res[basic_borders[0]:basic_borders[1]]
        char = self._batch_generator_class.vec2char(prediction, self._vocabulary)[0]
        if self._text_is_being_accumulated:
            if self._one_char_generation:
                self._accumulated_input += input_str
                self._accumulated_predictions += char
            else:
                self._accumulated_input.append(input_str)
                self._accumulated_predictions.append(char)
            self._accumulated_prob_vecs.append(np.reshape(prediction, [-1]))
        else:
            raise WrongMethodCallError('Flag self._accumulated_text should be set True when '
                                       'Handler._process_example_generation_results is called')
        if 'example_print_tensors' in self._last_run_tensor_order:
            print_borders = self._last_run_tensor_order['example_print_tensors']['borders']
            if print_borders[1] - print_borders[0] > 0:
                print_instructions = self._form_example_tensor_print_instructions(
                    step, char, res, self._last_run_tensor_order)
                self._print_tensors(
                    print_instructions,
                    print_step_number=True,
                    indent=1)

    def _process_several_launches_results(self, hp, results):
        self._environment_instance.append_to_storage(None, launches=(results, hp))
        self._save_launch_results(results, hp)
        self._print_launch_results(results, hp)

    def _process_several_optimizer_launches_results(self, hp, results):
        self._environment_instance.append_to_storage(None, launches=(results, hp))
        self._save_optimizer_launch_results(results, hp)

    def process_results(self, *args, regime=None):
        # print('in Handler.process_results')
        if regime == 'train':
            step = args[0]
            res = args[1]
            time_elapsed = args[2]
            # print("(Handler.process_results/train)self._save_path:", self._save_path)
            self._process_train_results(
                step, res, self._result_types, self._results_collect_interval, self._print_per_collected, time_elapsed)
        if regime == 'validation':
            step = args[0]
            res = args[1]
            self._process_validation_results(step, res)
        if regime == 'fuse':
            step = args[0]
            res = args[1]
            self._process_fuse_generation_results(step, res)
        if regime == 'example':
            step = args[0]
            input_str = args[1]
            res = args[2]
            self._process_example_generation_results(step, input_str, res)
        if regime == 'several_launches':
            hp = args[0]
            res = args[1]
            self._process_several_launches_results(hp, res)
        if regime == 'several_meta_optimizer_launches':
            hp = args[0]
            res = args[1]
            # print("(Handler.process_results/several_meta_optimizer_launches)self._save_path:", self._save_path)
            self._process_several_optimizer_launches_results(hp, res)
        if regime == 'validation_by_chars':
            step = args[0]
            res = args[1]
            tokens = args[2]
            self._process_validation_by_chars_results(step, res, tokens)
        if regime == 'train_meta_optimizer':
            step = args[0]
            res = args[1]
            # print("(Handler.process_results/train_optimizer)self._save_path:", self._save_path)
            # print("(Handler.process_results)self._results_collect_interval:", self._results_collect_interval)
            self._process_train_results(
                step, res, self._meta_optimizer_train_result_types,
                self._opt_train_results_collect_interval, self._opt_train_print_per_collected,
                msg='results on meta optimizer train op'
            )

    def log_launch(self):
        if self._save_path is None:
            if self._verbose:
                print('\nWarning! Launch is not logged because save_path was not provided to Handler constructor')
        else:
            self._current_log_path = add_index_to_filename_if_needed(self._save_path + '/launch_log.txt')
            with open(self._current_log_path, 'w') as f:
                now = dt.datetime.now()
                f.write(str(now) + '\n' * 2)
                f.write('launch regime: ' + self._processing_type + '\n' * 2)
                if self._processing_type == 'train' or self._processing_type == 'test':
                    f.write('build parameters:\n')
                    f.write(nested2string(self._environment_instance.current_pupil_build_parameters) + '\n' * 2)
                    f.write('launch parameters:\n')
                    f.write(nested2string(self._environment_instance.current_pupil_launch_parameters) + '\n' * 2)
                    f.write('default launch parameters:\n')
                    f.write(nested2string(
                        self._environment_instance.get_default_method_parameters( self._processing_type)) + '\n' * 2)
                elif self._processing_type == 'several_launches':
                    f.write('all_parameters:\n')
                    f.write(nested2string(self._environment_instance.current_pupil_launch_parameters) + '\n' * 2)
                    f.write('train method default parameters:\n')
                    f.write(nested2string(self._environment_instance.get_default_method_parameters('train')) + '\n' * 2)
                elif self._processing_type == 'train_with_meta':
                    f.write('pupil build parameters:\n')
                    f.write(nested2string(self._environment_instance.current_pupil_build_parameters) + '\n' * 2)
                    f.write('pupil launch parameters:\n')
                    f.write(nested2string(self._environment_instance.current_pupil_launch_parameters) + '\n' * 2)
                    f.write('default launch parameters:\n')
                    f.write(nested2string(
                        self._environment_instance.get_default_method_parameters('train')) + '\n' * 2)
                    f.write('optimizer build parameters:\n')
                    f.write(nested2string(
                        self._environment_instance.current_optimizer_build_parameters) + '\n' * 2)
                elif self._processing_type == 'train_meta_optimizer':
                    f.write('pupil build parameters:\n')
                    f.write(nested2string(self._environment_instance.current_pupil_build_parameters) + '\n' * 2)
                    f.write('optimizer launch parameters:\n')
                    f.write(nested2string(self._environment_instance.current_pupil_launch_parameters) + '\n' * 2)
                    f.write('default launch parameters:\n')
                    f.write(nested2string(
                        self._environment_instance.get_default_method_parameters('train_optimizer')) + '\n' * 2)
                    f.write('optimizer build parameters:\n')
                    f.write(nested2string(
                        self._environment_instance.current_optimizer_build_parameters) + '\n' * 2)
                elif self._processing_type == 'several_meta_optimizer_launches':
                    f.write('all_parameters:\n')
                    f.write(nested2string(self._environment_instance.current_optimizer_launch_parameters) + '\n' * 2)
                    f.write('train method default parameters:\n')
                    f.write(
                        nested2string(
                            self._environment_instance.get_default_method_parameters('train_optimizer')) + '\n' * 2
                    )

    def log_finish_time(self):
        if self._current_log_path is not None:
            with open(self._current_log_path, 'a') as f:
                now = dt.datetime.now()
                f.write('\nfinish time: ' + str(now) + '\n')

    def close(self):
        pass
