import os
import tensorflow as tf

from learning_to_learn.environment import Environment
from learning_to_learn.pupils.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary


voc_name = 'dost_voc.txt'
with open(voc_name, 'r') as f:
    vocabulary = list(f.read())

vocabulary_size = len(vocabulary)

env = Environment(Lstm, BatchGenerator, vocabulary=vocabulary)

cpiv = get_positions_in_vocabulary(vocabulary)

add_feed = [{'placeholder': 'dropout', 'value': 0.9} #,
            # {'placeholder': 'sampling_prob',
            #  'value': {'type': 'linear', 'start': 0., 'end': 1., 'interval': 3000}},
            # {'placeholder': 'loss_comp_prob',
            #  'value': {'type': 'linear', 'start': 1., 'end': 0., 'interval': 3000}}
            ]
valid_add_feed = [# {'placeholder': 'sampling_prob', 'value': 1.},
                  {'placeholder': 'dropout', 'value': 1.}]

add_metrics = ['bpc', 'perplexity', 'accuracy']

tf.set_random_seed(1)

NUM_UNROLLINGS = 100
BATCH_SIZE = 32
env.build_pupil(
    batch_size=BATCH_SIZE,
    num_layers=2,
    num_nodes=[2000, 2000],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=1000,
    num_gpus=1,
    regime='inference',
)

print('building is finished')
stop_specs = dict(
    type='while_progress',
    max_no_progress_points=10,
    changing_parameter_name='learning_rate',
    path_to_target_metric_storage=('default_1', 'loss')
)
learning_rate = dict(
    type='adaptive_change',
    max_no_progress_points=10,
    decay=.5,
    init=4.,
    path_to_target_metric_storage=('default_1', 'loss')
)
env.test(
    restore_path='dostoevsky/train/checkpoints/best',
    save_path='lstm/text8_pretrain/validation200',
    vocabulary=vocabulary,
    additions_to_feed_dict=valid_add_feed,
    validation_dataset_texts=[valid_text],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy']
)
