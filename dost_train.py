import os
import tensorflow as tf

from learning_to_learn.environment import Environment
from learning_to_learn.pupils.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary

with open('../../skorokhodov/neuro_dostoevsky/data/subs/subs.ru', 'r') as f:
    text = f.read()
# with open('datasets/text8.txt', 'r') as f:
#     text = f.read()

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]

voc_name = 'dost_voc.txt'
if os.path.isfile(voc_name):
    with open(voc_name, 'r') as f:
        vocabulary = list(f.read())
else:
    vocabulary = create_vocabulary(train_text + valid_text)
    with open(voc_name, 'w') as f:
        f.write(''.join(vocabulary))

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
    num_unrollings=NUM_UNROLLINGS,
    init_parameter=3.,
    # character_positions_in_vocabulary=cpiv,
    num_gpus=1,
    additional_metrics=add_metrics,
    going_to_limit_memory=True,
    optimizer='sgd'
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
env.train(
    # gpu_memory=.3,
    allow_growth=True,
    save_path='lstm/start',
    # restore_path='lstm_sample_test/scipop3_1000_bs256_11.12/checkpoints/2000',
    learning_rate=learning_rate,
    batch_size=BATCH_SIZE,
    num_unrollings=NUM_UNROLLINGS,
    vocabulary=vocabulary,
    checkpoint_steps=None,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=stop_specs,
    # stop=4000,
    train_dataset_text=train_text,
    # train_dataset_text='abc',
    validation_dataset_texts=[valid_text],
    results_collect_interval=1000,
    additions_to_feed_dict=add_feed,
    validation_additions_to_feed_dict=valid_add_feed,
    no_validation=False
)
