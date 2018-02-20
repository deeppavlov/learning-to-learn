import re
import tensorflow as tf
from environment import Environment
# from gru_par import Gru, BatchGenerator
# from lstm_sample_par import Lstm, LstmBatchGenerator
from lstm_par import Lstm, LstmBatchGenerator
# from lstm_par import Lstm, LstmFastBatchGenerator
from some_useful_functions import create_vocabulary, get_positions_in_vocabulary, construct

# f = open('datasets/scipop_v3.0/scipop_train.txt', 'r', encoding='utf-8')
# text = f.read()
# text = re.sub('<[^>]*>', '', text)
# f.close()

# from bpe import BpeBatchGenerator, BpeBatchGeneratorOneHot, create_vocabulary
# from ngrams import NgramsBatchGenerator, create_vocabulary
# from some_useful_functions import get_positions_in_vocabulary
# f = open('datasets/scipop_v3.0/bpe_train.txt', 'r', encoding='utf-8')
f = open('datasets/scipop_v3.0/scipop_train.txt', 'r', encoding='utf-8')
text = f.read()
f.close()

text = re.sub('<[^>]*>', '', text)
text = re.sub('\xa0', ' ', text)
# different
offset = 190
valid_size = 10000
valid_text = text[offset:offset + valid_size]
# train_text = text[offset + valid_size:]
train_text = text[offset:]
print('(__main__)valid_text:', valid_text)
train_size = len(train_text)

# In[5]:

vocabulary = create_vocabulary(text)
# print('vocabulary:', vocabulary)
print('len(vocabulary):', len(vocabulary))
vocabulary_size = len(vocabulary)

# """lstm"""
# env = Environment(Lstm, LstmBatchGenerator, vocabulary=vocabulary)
#
cpiv = get_positions_in_vocabulary(vocabulary)


# tensor_names = [('mask', 'validation/iter_0/force_or_sample/random_modifier:0')]
# tensor_names = [('mask', 'validation/sample_input:0')]
# tensor_names = [('in_s_flag', 'validation/sample_input_and_in_s_flag:1')]
# tensor_names.append(('mask', 'validation/iter_0/force_or_sample/mask:0'))
# tensor_names.append(('answer', 'validation/iter_0/force_or_sample/sampled_answer:0'))
# tensor_names.append(('after_choosing', 'validation/iter_0/force_or_sample/inp_after_choosing:0'))
# tensor_names.append(('final_dev_out_s_flags', 'train/gpu0/device_gradient_computation/final_dev_out_s_flags:0'))
# tensor_names.append(('number_of_computed_losses', 'train/cpu0_gradients/number_of_computed_losses:0'))
# tensor_names.append(('out_s_on_dev_0', 'out_s_on_dev_0:0'))
# for i in range(3):
#     tensor_names.append(('tr_in_s_flag_%s' % i, 'in_s_flags_on_dev_0:%s' % i))
#     tensor_names.append(('tr_mask_%s' % i, 'train/gpu0/iter_%s/force_or_sample/mask:0' % i))
#     tensor_names.append(('tr_answer_%s' % i, 'train/gpu0/iter_%s/force_or_sample/sampled_answer:0' % i))
#     tensor_names.append(('tr_after_choosing_%s' % i, 'train/gpu0/iter_%s/force_or_sample/inp_after_choosing:0' % i))
#     # tensor_names.append(('tr_predictions_%s' % i,
#     #                      'train/gpu0/device_gradient_computation/predictions_%s:0' % i))
#     tensor_names.append(('tr_input_%s' % i,
#                          'inp_on_dev_0:%s' % i))
#
# valid_tensors = {'valid_print_tensors': {'mask': [100 + i for i in range(30)],
#                                          'in_s_flag': [100 + i for i in range(30)],
#                                          'answer': [100 + i for i in range(30)],
#                                          'after_choosing': [100 + i for i in range(30)],
#                                          'sample_input': [100 + i for i in range(30)]}}
#
# train_print = dict()
# train_print['number_of_computed_losses'] = [i for i in range(20)]
# # train_print['out_s_on_dev_0'] = [i for i in range(20)]
# # train_print['final_dev_out_s_flags'] = [i for i in range(20)]
# for idx in range(3):
#     pass
#     # train_print['tr_in_s_flag_%s' % idx] = [i for i in range(20)]
#     # train_print['tr_mask_%s' % idx] = [i for i in range(20)]
#     # train_print['tr_answer_%s' % idx] = [i for i in range(20)]
#     # train_print['tr_after_choosing_%s' % idx] = [i for i in range(20)]
#     # train_print['tr_input_%s' % idx] = [i for i in range(20)]
#     # train_print['tr_predictions_%s' % idx] = [i for i in range(20)]
# train_tensors = {'train_print_tensors': train_print}
#
#
# def all_non_zero(**kwargs):
#     tensors = list()
#     for key, value in kwargs.items():
#         if key != 'special_args':
#             tensors.append(value)
#     return sum([tf.reduce_sum(t) for t in tensors])
#
#
# def in_and_out_f_comp(**kwargs):
#     kwargs = dict(kwargs.items())
#     out_fs = kwargs['out']
#     del kwargs['out']
#     del kwargs['special_args']
#     in_fs = list()
#     for k, v in sorted(kwargs.items(), key=lambda item: int(item[0][3:])):
#         # print('building in_fs:')
#         if 'in' in k:
#             print(k)
#             in_fs.append(v)
#     out_shape = out_fs.get_shape().as_list()
#     num_unrollings = len(in_fs)
#     bsize = out_shape[0] // num_unrollings
#     out_fs = tf.reshape(out_fs, [num_unrollings, bsize, 1])
#     one_chunk_out = tf.reshape(tf.slice(out_fs, [0, 0, 0], [num_unrollings, 1, 1]), [-1])
#     in_all = tf.stack(in_fs)
#     one_chunk_in = tf.reshape(tf.slice(in_all, [0, 0, 0], [num_unrollings, 1, 1]), [-1])
#     return tf.stack([one_chunk_in, one_chunk_out], axis=1)
#
# env.register_build_function(all_non_zero, 'all_nz')
# env.register_build_function(in_and_out_f_comp, 'in_and_out_flags')
#
#
# def all_nz(tmpl, hook_name, number):
#     schedule = dict()
#     env.register_builder('all_nz',
#                          tensor_names=dict([('tensor_%s' % i, tmpl % i) for i in range(number)]),
#                          output_hook_name=hook_name)
#     schedule[hook_name] = [i for i in range(30)]
#     return schedule
#
#
# def in_and_out(hook_name, number):
#     schedule = dict()
#     tensor_names = dict()
#     tensor_names['out'] = 'out_s_on_dev_0:0'
#     for i in range(number):
#         tensor_names['in_%s' % i] = 'in_s_flags_on_dev_0:%s' % i
#     env.register_builder('in_and_out_flags',
#                          tensor_names=tensor_names,
#                          output_hook_name=hook_name)
#     schedule[hook_name] = [i for i in range(30)]
#     return schedule
#
# # train_tensors['train_print_tensors'].update(all_nz('in_s_flags_on_dev_0:%s', 'nz_in_flags', 20))
# # train_tensors['train_print_tensors'].update(all_nz('out_s_on_dev_0:%s', 'nz_out_flags', 1))
# train_tensors['train_print_tensors'].update(in_and_out('in_and_out_flags', 20))
#
# """lstm sample"""
# add_feed = [{'placeholder': 'dropout', 'value': 0.9},
#             {'placeholder': 'sampling_prob',
#              'value': {'type': 'linear', 'start': 0., 'end': 1., 'interval': 3000}},
#             {'placeholder': 'loss_comp_prob',
#              'value': {'type': 'linear', 'start': 1., 'end': 0., 'interval': 3000}}]
# valid_add_feed = [# {'placeholder': 'sampling_prob', 'value': 1.},
#                   {'placeholder': 'dropout', 'value': 1.}]
# env.build(batch_size=64,
#           num_layers=2,
#           num_nodes=[100, 100],
#           num_output_layers=2,
#           num_output_nodes=[124],
#           vocabulary_size=vocabulary_size,
#           embedding_size=128,
#           num_unrollings=10,
#           character_positions_in_vocabulary=cpiv)
#
# env.add_hooks(tensor_names=tensor_names)
# env.train(save_path='debugging_lstm_sample/first',
#           allow_growth=False,
#           gpu_memory=.5,
#           # log_device_placement=True,
#           learning_rate={'type': 'exponential_decay',
#                          'init': .001,
#                          'decay': .9,
#                          'period': 2000},
#           additions_to_feed_dict=add_feed,
#           validation_additions_to_feed_dict=valid_add_feed,
#           batch_size=64,
#           num_unrollings=10,
#           vocabulary=vocabulary,
#           checkpoint_steps=[100],
#           result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#           printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#           # validation_tensor_schedule=valid_tensors,
#           # train_tensor_schedule=train_tensors,
#           stop=5000,
#           # train_dataset_text='abx',
#           # validation_datasets_texts=['abc'],
#           train_dataset_text=train_text,
#           validation_dataset_texts=[valid_text],
#           # validation_dataset=[valid_text],
#           results_collect_interval=200,
#           no_validation=False,
#           add_graph_to_summary=True)

"""lstm and gru"""
# env = Environment(Gru, BatchGenerator, vocabulary=vocabulary)
env = Environment(Lstm, LstmBatchGenerator, vocabulary=vocabulary)
# env = Environment(Lstm, LstmFastBatchGenerator, vocabulary=vocabulary)
# env = Environment(Lstm, BpeBatchGenerator, vocabulary=vocabulary)
# env = Environment(Lstm, NgramsBatchGenerator, vocabulary=vocabulary)
add_feed = [{'placeholder': 'dropout', 'value': 0.9}]
valid_add_feed = [{'placeholder': 'dropout', 'value': 1.}]

env.build(batch_size=64,
          # embeddings_in_batch=False,
          num_layers=2,
          num_nodes=[300, 300],
          num_output_layers=2,
          num_output_nodes=[124],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=20,
          going_to_limit_memory=True)

# env.add_hooks(tensor_names=tensor_names)
env.train(save_path='debugging_bpe/first',
          # restore_path='debugging_bpe/first/checkpoints/30000',
          learning_rate={'type': 'exponential_decay',
                         'init': .00,
                         'decay': .2,
                         'period': 30000},
          additions_to_feed_dict=add_feed,
          validation_additions_to_feed_dict=valid_add_feed,
          # validate_tokens_by_chars=True,
          batch_size=64,
          num_unrollings=20,
          vocabulary=vocabulary,
          checkpoint_steps=10000,
          result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
          # validation_tensor_schedule=valid_tensors,
          # train_tensor_schedule=train_tensors,
          stop=30001,
          # train_dataset_text='abx',
          # validation_datasets_texts=['abc'],
          train_dataset_text=train_text,
          validation_dataset_texts=[valid_text],
          # validation_dataset=[valid_text],
          results_collect_interval=2000,
          no_validation=False,
          gpu_memory=.5)

# connection_interval = 8
# connection_visibility = 5
# subsequence_length_in_intervals = 10

# env.build(batch_size=64,
#           num_layers=2,
#           num_nodes=[200, 200],
#           num_output_layers=2,
#           num_output_nodes=[200],
#           vocabulary_size=vocabulary_size,
#           embedding_size=128,
#           #dim_compressed=4,
#           connection_interval=connection_interval,
#           subsequence_length_in_intervals=subsequence_length_in_intervals,
#           connection_visibility=connection_visibility)

# tensor_names = [('sample_state_0', 'validation/saved_sample_from_connections_0:0'),
#                 ('for_conn', 'validation/saved_sample_state:0')]
# for idx in range(connection_visibility):
#     tensor_names.append(('saved_for_connection_%s' % idx, 'train/saved_for_connection_%s:0' % idx))
# env.add_hooks(tensor_names=tensor_names)
#
# valid_tensors = {'valid_print_tensors': {'sample_state_0': [i for i in range(100)],
#                                          'for_conn': [i for i in range(100)]}}
# train_print = dict()
# for idx in range(connection_visibility):
#     train_print['saved_for_connection_%s' % idx] = [i for i in range(100)]
# train_tensors = {'train_print_tensors': train_print}

# env.train(save_path='debugging_environment/first',
#           learning_rate={'type': 'exponential_decay',
#                          'init': .002,
#                          'decay': .5,
#                          'period': 20000},
#           additions_to_feed_dict=add_feed,
#           validation_additions_to_feed_dict=valid_add_feed,
#           batch_size=64,
#           num_unrollings=subsequence_length_in_intervals*connection_interval,
#           vocabulary=vocabulary,
#           checkpoint_steps=1000,
#           result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#           printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#           stop=10000,
#           #validation_tensor_schedule=valid_tensors,
#           #train_tensor_schedule=train_tensors,
#           #train_dataset_text='abx',
#           #validation_datasets_texts=['abc'],
#           train_dataset_text=train_text,
#           validation_dataset_texts=[valid_text],
#           #validation_dataset=[valid_text],
#           results_collect_interval=1000)

# evaluation = dict(
#     save_path='simple_lstm/tuning2',
#     result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#     datasets={'train': None,
#               'default_1': [valid_text, 'default_1']},
#     batch_gen_class=BatchGenerator,
#     batch_kwargs={'vocabulary': vocabulary},
#     batch_size=1,
#     additions_to_feed_dict=[{'type': 'fixed', 'name': 'dropout', 'value': 1.}]
# )
#
# kwargs_for_building = dict(
#           batch_size=64,
#           num_layers=2,
#           num_nodes=[300, 300],
#           vocabulary_size=vocabulary_size,
#           num_unrollings=30)
#
# #list_of_lr = [dict(type='exponential_decay', init=v, decay=.9, period=500) for v in [10., 5., 3., 1., .3]]
#
# env.grid_search(evaluation,
#                      kwargs_for_building,
#                      #build_hyperparameters={'init_parameter': [.01, .03, .1, .3, 1., 3.]},
#                      build_hyperparameters={'num_nodes[0,1]': [[250, 350, 450]]*2},
#                      # other_hyperparameters={'learning_rate': {'hp_type': 'built-in',
#                      #                                          'type': 'exponential_decay',
#                      #                                          'fixed': {'period': 500, 'decay': .9},
#                      #                                          'varying': {'init': [.002, .001, .0015]}}},
#                      batch_size=64,
#                      result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#                      vocabulary=vocabulary,
#                      stop=50,
#                      num_unrollings=30,
#                      train_dataset_text=train_text,
#                      printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
#                      results_collect_interval=25,
#                      validation_dataset_texts=[valid_text],
#                      no_validation=False,
#                      additions_to_feed_dict=[{'type': 'fixed', 'name': 'dropout', 'value': 1.}])

# env.build(batch_size=64,
#           num_layers=2,
#           num_nodes=[200, 200],
#           num_output_layers=2,
#           num_output_nodes=[200],
#           vocabulary_size=vocabulary_size,
#           embedding_size=128,
#           #dim_compressed=4,
#           connection_interval=connection_interval,
#           subsequence_length_in_intervals=subsequence_length_in_intervals,
#           connection_visibility=connection_visibility,
#           regime='inference')

# env.build(batch_size=1,
#           num_layers=2,
#           num_nodes=[300, 300],
#           num_output_layers=2,
#           num_output_nodes=[124],
#           vocabulary_size=vocabulary_size,
#           embedding_size=128,
#           num_unrollings=1,
#           going_to_limit_memory=True)
#
# env.inference(restore_path='debugging_bpe/first/checkpoints/final',
#               log_path='debugging_bpe/first/debug_split_inference',
#               batch_generator_class=BpeBatchGenerator,
#               character_positions_in_vocabulary=cpiv,
#               vocabulary=vocabulary,
#               additions_to_feed_dict=[{'placeholder': 'dropout', 'value': 1.}],
#               gpu_memory=.1,
#               bpe_codes='datasets/scipop_v3.0/codes.txt')

# env.generate_discriminator_dataset(10000000, 1, text, 200, 'new_line', 'debugging_environment/first/checkpoints/1000',
#                                    'debugging_environment/first/debug_gen_dataset',
#                                    additions_to_feed_dict=[{'placeholder': 'dropout', 'value': 1.}], gpu_memory=.1)