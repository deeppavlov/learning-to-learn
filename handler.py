import tensorflow as tf
import numpy as np
import datetime as dt
# import os
from useful_functions import create_path, add_index_to_filename_if_needed, construct, nested2string, \
    WrongMethodCallError

class Handler(object):

    _stars = '*'*30

    def __init__(self,
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
                 hyperparameters=None,
                 # test method specific
                 validation_dataset_names=None,
                 validation_tensor_schedule=None,
                 printed_result_types=None,
                 fuses=None,
                 fuse_tensor_schedule=None,
                 fuse_file_name=None,
                 example_tensor_schedule=None,
                 example_file_name=None,
                 verbose=True):
        self._verbose = verbose
        if printed_result_types is None:
            printed_result_types = ['loss']
        continuous_chit_chat = ['simple_fontain']
        # print('Initializing Handler! pid = %s' % os.getpid())
        self._processing_type = processing_type
        self._environment_instance = environment_instance
        self._save_path = save_path
        self._current_log_path = None
        self._result_types = self._environment_instance.put_result_types_in_correct_order(result_types)
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

        if self._processing_type == 'train':
            self._train_files = dict()
            if self._save_path is not None:
                self._train_files['loss'] = self._save_path + '/' + 'loss_train.txt'
                self._train_files['perplexity'] = self._save_path + '/' + 'perplexity_train.txt'
                self._train_files['accuracy'] = self._save_path + '/' + 'accuracy_train.txt'
                if self._bpc:
                    self._train_files['bpc'] = self._save_path + '/' + 'bpc_train.txt'
                self._train_files['pickle_tensors'] = self._save_path + '/' + 'tensors_train.pickle'
            self._train_dataset_name = None
            self._dataset_specific = dict()
            self._controllers = None
            self._results_collect_interval = None
            self._print_per_collected = None
            self._example_per_print = None
            self._train_tensor_schedule = None
            self._validation_tensor_schedule = None

            self._fuses = None
            self._print_fuses = True
            self._fuse_file_name = None
            self._fuse_tensor_schedule = None

            self._print_examples = True
            self._example_file_name = None
            self._example_tensor_schedule = None

            self._processed_fuse_index = None

            self._text_is_being_accumulated = False
            self._accumulated_text = None
            self._accumulated_input = None
            self._accumulated_predictions = None
            self._accumulated_prob_vecs = None

            self._printed_result_types = None
            self._printed_controllers = None
            if summary and self._save_path is not None:
                self._writer = tf.summary.FileWriter(self._save_path + '/' + 'summary')
                if add_graph_to_summary:
                    self._writer.add_graph(tf.get_default_graph())
            self._environment_instance.init_storage('train',
                                                    steps=list(),
                                                    loss=list(),
                                                    perplexity=list(),
                                                    accuracy=list(),
                                                    bpc=list())
            self._training_step = None
            self._accumulation_is_performed = False
            self._accumulated_tensors = dict()
            self._accumulated = dict(loss=None, perplexity=None, accuracy=None)
            if self._bpc:
                self._accumulated['bpc'] = None
        if self._processing_type == 'test':
            self._name_of_dataset_on_which_accumulating = None
            self._dataset_specific = dict()
            self._validation_tensor_schedule = validation_tensor_schedule
            self._validation_dataset_names = validation_dataset_names
            self._switch_datasets(validation_dataset_names=self._validation_dataset_names)
            self._printed_result_types = printed_result_types
            self._accumulated_tensors = dict(
                valid_print_tensors=dict(),
                valid_save_text_tensors=dict()
            )
            self._accumulated = dict(loss=None, perplexity=None, accuracy=None)
            if self._bpc:
                self._accumulated['bpc'] = None
            self._fuses = fuses
            if self._fuses is not None:
                for fuse in self._fuses:
                    fuse['results'] = list()
            if fuse_file_name is not None:
                self._fuse_file_name = fuse_file_name
            elif self._fuses is not None and self._save_path is not None:
                self._fuse_file_name = add_index_to_filename_if_needed(self._save_path + '/fuses.txt')
            self._fuse_tensor_schedule = fuse_tensor_schedule

            self._processed_fuse_index = None

            if example_file_name is not None:
                self._example_file_name = example_file_name
            elif self._save_path is not None:
                self._example_file_name = add_index_to_filename_if_needed(self._save_path + '/examples.txt')
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

        if self._processing_type == 'several_launches':
            self._result_types = result_types
            self._eval_dataset_names = eval_dataset_names
            self._save_path = save_path
            self._environment_instance = environment_instance
            self._hyperparameters = hyperparameters
            self._order = list(self._result_types)
            if self._hyperparameters is not None:
                self._order.extend(self._hyperparameters)

            create_path(self._save_path)
            self._file_names = dict()
            self._tmpl = '%s '*(len(self._order) - 1) + '%s\n'
            result_names = list()
            for result_type in self._order:
                if not isinstance(result_type, tuple):
                    result_names.append(result_type)
                else:
                    result_names.append(self._hyperparameter_name_string(result_type))
            for dataset_name in eval_dataset_names:
                self._file_names[dataset_name] = self._save_path + '/' + dataset_name + '.txt'
                fd = open(self._file_names[dataset_name],
                          'a')
                fd.write(self._tmpl % tuple(result_names))
                fd.close()
            self._environment_instance.set_in_storage(launches=list())

        # The order in which tensors are presented in the list returned by get_additional_tensors method
        # It is a list. Each element is either tensor alias or a tuple if corresponding hook is pointing to a list of
        # tensors. Such tuple contains tensor alias, and sizes of nested lists

    def _switch_datasets(self, train_dataset_name=None, validation_dataset_names=None):
        if train_dataset_name is not None:
            self._train_dataset_name = train_dataset_name
        if validation_dataset_names is not None:
            #print('validation_dataset_names:', validation_dataset_names)
            #print('env._storage:', self._environment_instance._storage)
            for dataset_name in validation_dataset_names:
                if dataset_name not in self._dataset_specific.keys():
                    new_files = dict()
                    if self._save_path is not None:
                        new_files['loss'] = (self._save_path + '/' + 'loss_validation_%s.txt' % dataset_name)
                        new_files['perplexity'] = (self._save_path + '/' +
                                                   'perplexity_validation_%s.txt' % dataset_name)
                        new_files['accuracy'] = (self._save_path + '/' + 'accuracy_validation_%s.txt' % dataset_name)
                        if self._bpc:
                            new_files['bpc'] = (self._save_path + '/' + 'bpc_validation_%s.txt' % dataset_name)
                        new_files['pickle_tensors'] = (self._save_path + '/' +
                                                       'tensors_validation_%s.pickle' % dataset_name)

                    self._dataset_specific[dataset_name] = {'name': dataset_name,
                                                            'files': new_files}
                    init_dict = dict()
                    for key in self._result_types:
                        if not self._environment_instance.check_if_key_in_storage([dataset_name, key]):
                            init_dict[key] = list()
                    #print('dataset_name:', dataset_name)
                    self._environment_instance.init_storage(dataset_name, **init_dict)

    def set_new_run_schedule(self, schedule, train_dataset_name, validation_dataset_names):
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
        self._printed_controllers = schedule['printed_controllers']
        self._printed_result_types = schedule['printed_result_types']

        self._fuses = schedule['fuses']
        self._fuse_tensor_schedule = schedule['fuse_tensors']
        if self._fuses is not None:
            for fuse in self._fuses:
                fuse['results'] = list()
        if self._fuse_file_name is None and self._fuses is not None and self._save_path is not None:
            self._fuse_file_name = add_index_to_filename_if_needed(self._save_path + '/fuses.txt')

        self._example_tensor_schedule = schedule['example_tensors']

        if self._example_file_name is None and self._save_path is not None:
            self._example_file_name = add_index_to_filename_if_needed(self._save_path + '/examples.txt')

        if self._printed_result_types is not None:
            if len(self._printed_result_types) > 0:
                self._print_results = True
        self._switch_datasets(train_dataset_name, validation_dataset_names)

    def set_test_specs(self, validation_dataset_names=None, fuses=None, replicas=None, random=None):
        if validation_dataset_names is not None:
            self._switch_datasets(validation_dataset_names=validation_dataset_names)

    def set_controllers(self, controllers):
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

    def stop_accumulation(self,
                          save_to_file=True,
                          save_to_storage=True,
                          print_results=True):
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
                    file_name = self._dataset_specific[self._name_of_dataset_on_which_accumulating]['files'][key]
                    with open(file_name, 'a') as f:
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
            self._environment_instance.append_to_storage(self._name_of_dataset_on_which_accumulating,
                **dict([(key, means[key]) for key in self._result_types]))
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
        with open(self._fuse_file_name, 'a', encoding='utf-8') as f:
            f.write(self._form_fuse_msg(training_step) + '\n'*2)

    def _print_example_results(self, training_step):
        print(self._form_example_msg(training_step))

    def _save_example_results(self, training_step):
        with open(self._example_file_name, 'a', encoding='utf-8') as f:
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
        if self._bpc:
            [loss, perplexity, accuracy, bpc] = tmp_output
            self._accumulate_several_data(['loss', 'perplexity', 'accuracy', 'bpc'], [loss, perplexity, accuracy, bpc])
        else:
            [loss, perplexity, accuracy] = tmp_output
            self._accumulate_several_data(['loss', 'perplexity', 'accuracy'], [loss, perplexity, accuracy])
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
        if not isinstance(self._hooks[alias], list):
            return [self._hooks[alias]], 1
        add_tensors = list()
        order = [alias, len(self._hooks[alias])]
        counter = 0
        if isinstance(self._hooks[alias][0], list):
            order.append(len(self._hooks[alias][0]))
            for elem in self._hooks[alias]:
                for tensor in elem:
                    add_tensors.append(tensor)
                    counter += 1
        else:
            for tensor in self._hooks[alias]:
                add_tensors.append(tensor)
                counter += 1
        return add_tensors, counter

    def _save_datum(self, descriptor, step, datum, processing_type, dataset_name):
        if processing_type == 'train':
            file_name = self._train_files[descriptor]
            # print('file_name:', file_name)
            # print('self._train_files:', self._train_files)
            with open(file_name, 'a') as f:
                f.write('%s %s\n' % (step, datum))
        elif processing_type == 'validation':
            file_name = self._dataset_specific[dataset_name]['files'][descriptor]
            with open(file_name, 'a') as f:
                f.write('%s %s\n' % (step, datum))

    def _save_launch_results(self, results, hp):
        for dataset_name, res in results.items():
            values = list()
            all_together = dict(hp.items())
            #print('dataset_name:', dataset_name)
            #print('all_together:', all_together)
            all_together.update(res)
            for key in self._order:
                values.append(all_together[key])
            with open(self._file_names[dataset_name], 'a') as f:
                f.write(self._tmpl % tuple(values))

    def _save_several_data(self,
                           descriptors,
                           step,
                           data,
                           processing_type='train',
                           dataset_name=None):
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
                tensors.append(self._hooks['train_op_with_meta_optimizer'])
                current['tensors']['train_op_with_meta_optimizer'] = [pointer, pointer+1]
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
        # print(tensors)
        return tensors

    def _get_additional_tensors(self,
                                schedule,
                                step,
                                start_pointer):
        #print('_get_additional_tensors method. schedule:', schedule)
        additional_tensors = list()
        pointer = start_pointer
        for tensors_use, tensors_schedule in schedule.items():
            #print('tensor_schedule:', tensors_schedule)
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

    def _print_tensors(self, instructions, print_step_number=False, indent=0):
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
    def _hyperparameter_name_string(name):
        # print('(Handler._hyperparameter_name_string)name:', name)
        string = name[1]
        if name[2] is not None:
            string += '[%s]' % name[2]
        if name[3] is not None:
            string += '/' + name[3]
        return string

    def _print_launch_results(self, results, hp, idx=None, indent=2):
        if indent != 0:
            print('\n' * (indent - 1))
        for key in self._order:
            if key in hp:
                print('%s: %s' % (self._hyperparameter_name_string(key), hp[key]))
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
                #     print(controller.name)
                # if isinstance(controller, list):
                #     for c in controller:
                #         if isinstance(c, Controller):
                #             print(c.name)
                #         else:
                #             print(c)
                #print('controller._specifications:', controller._specifications)
                if controller.name in self._printed_controllers:
                    print('%s:' % controller.name, controller.get())

    def _print_standard_report(self,
                               indents=[0, 0],
                               regime='train',
                               **kwargs):
        for _ in range(indents[0]):
            print('')
        if regime == 'train':
            if 'step' in kwargs:
                print('step:', kwargs['step'])
            self._print_controllers()
        if 'message' in kwargs:
            print(kwargs['message'])
        for key, value in kwargs.items():
            if key != 'tensors' and\
                            key != 'step' and \
                            key != 'message' and \
                            key in self._printed_result_types and \
                            key not in self._print_order:
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
                    extracted[alias] = res[borders[0], borders[1]]
                else:
                    structured = list()
                    pointer = borders[0]
                    for length in structure[1:]:
                        structured.append(res[pointer, pointer+length])
                    extracted[alias] = structured
        return extracted

    def _form_train_tensor_print_instructions(self, step, train_res, last_order):
        instructions = dict()
        instructions['step'] = step
        instructions['message'] = 'train tensors:'
        instructions['results'] = dict()
        extracted_for_printing = self._extract_results(last_order, 'train_print_tensors', train_res)
        #print('extracted_for_printing:', extracted_for_printing)
        instructions['results'].update(extracted_for_printing)
        return instructions

    def _process_train_results(self,
                               step,
                               train_res):
        # print('step:', step)
        # print('train_res:', train_res)
        #print(self._last_run_tensor_order)
        basic_borders = self._last_run_tensor_order['basic']['borders']
        tmp = train_res[basic_borders[0]+1:basic_borders[1]]
        if self._bpc:
            [loss, perplexity, accuracy, bpc] = tmp
        else:
            [loss, perplexity, accuracy] = tmp

        if self._printed_result_types is not None:
            if self._results_collect_interval is not None:
                if step % (self._results_collect_interval * self._print_per_collected) == 0:
                    if self._bpc:
                        self._print_standard_report(indents=[2, 0],
                                                    step=step,
                                                    loss=loss,
                                                    bpc=bpc,
                                                    perplexity=perplexity,
                                                    accuracy=accuracy,
                                                    message='results on train dataset')
                    else:
                        self._print_standard_report(indents=[2, 0],
                                                    step=step,
                                                    loss=loss,
                                                    perplexity=perplexity,
                                                    accuracy=accuracy,
                                                    message='results on train dataset')
        if self._results_collect_interval is not None:
            if step % self._results_collect_interval == 0:
                if self._bpc:
                    if self._save_path is not None:
                        self._save_several_data(['loss', 'perplexity', 'accuracy', 'bpc'],
                                                step,
                                                [loss, perplexity, accuracy, bpc])
                    self._environment_instance.append_to_storage('train',
                                                                 loss=loss,
                                                                 bpc=bpc,
                                                                 perplexity=perplexity,
                                                                 accuracy=accuracy,
                                                                 steps=step)
                else:
                    if self._save_path is not None:
                        self._save_several_data(['loss', 'perplexity', 'accuracy'], step, [loss, perplexity, accuracy])
                    self._environment_instance.append_to_storage('train',
                                                                 loss=loss,
                                                                 perplexity=perplexity,
                                                                 accuracy=accuracy,
                                                                 steps=step)
        print_borders = self._last_run_tensor_order['train_print_tensors']['borders']
        if print_borders[1] - print_borders[0] > 0:
            print_instructions = self._form_train_tensor_print_instructions(step,
                                                                            train_res,
                                                                            self._last_run_tensor_order)
            other_stuff_is_printed = (step % (self._results_collect_interval * self._print_per_collected) == 0)
            if other_stuff_is_printed:
                indent = 0
            else:
                indent = 1
            self._print_tensors(print_instructions,
                                print_step_number=not other_stuff_is_printed,
                                indent=indent)

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
        #print('extracted_for_printing:', extracted_for_printing)
        instructions['results'].update(extracted_for_printing)
        return instructions

    def _form_example_tensor_print_instructions(self, step, char, example_res, last_order):
        instructions = dict()
        instructions['step'] = step
        instructions['message'] = 'example tensors:\nchar = %s' % self._form_string_char(char)
        instructions['results'] = dict()
        extracted_for_printing = self._extract_results(last_order, 'example_print_tensors', example_res)
        #print('extracted_for_printing:', extracted_for_printing)
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

    def _several_launches_results_processing(self, hp, results):
        self._environment_instance.append_to_storage(None, launches=(results, hp))
        self._save_launch_results(results, hp)
        self._print_launch_results(results, hp)

    def process_results(self, *args, regime=None):
        # print('in Handler.process_results')
        if regime == 'train':
            step = args[0]
            res = args[1]
            self._process_train_results(step, res)
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
        if regime =='several_launches':
            hp = args[0]
            res = args[1]
            self._several_launches_results_processing(hp, res)
        if regime == 'validation_by_chars':
            step = args[0]
            res = args[1]
            tokens = args[2]
            self._process_validation_by_chars_results(step, res, tokens)

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
                    f.write(nested2string(self._environment_instance.current_build_parameters) + '\n' * 2)
                    f.write('user specified parameters:\n')
                    f.write(nested2string(self._environment_instance.current_launch_parameters) + '\n' * 2)
                    f.write('default parameters:\n')
                    f.write(nested2string(
                        self._environment_instance.get_default_method_parameters( self._processing_type)) + '\n' * 2)
                elif self._processing_type == 'several_launches':
                    f.write('all_parameters:\n')
                    f.write(nested2string(self._environment_instance.current_launch_parameters) + '\n' * 2)
                    f.write('train method default parameters:\n')
                    f.write(nested2string(self._environment_instance.get_default_method_parameters('train')) + '\n' * 2)

    def log_finish_time(self):
        if self._current_log_path is not None:
            with open(self._current_log_path, 'a') as f:
                now = dt.datetime.now()
                f.write('\nfinish time: ' + str(now) + '\n')

    def close(self):
        pass
