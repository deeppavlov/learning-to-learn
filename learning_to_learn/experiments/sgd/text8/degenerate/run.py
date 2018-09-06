ROOT_HEIGHT = 5
import sys
from pathlib import Path
import tensorflow as tf
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[ROOT_HEIGHT]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.environment import Environment
from learning_to_learn.pupils.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from learning_to_learn.useful_functions import create_vocabulary
from learning_to_learn.launch_helpers import load_text_dataset

import os

conf_file = sys.argv[1]
save_path = os.path.join(conf_file.split('.')[0], 'results')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

with open(conf_file, 'r') as f:
    lines = f.read().split('\n')

dataset_path = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'text8.txt']))
with open(dataset_path, 'r') as f:
    text = f.read()

BATCH_SIZE = 32
NUM_UNROLLINGS = 10
RESULTS_COLLECT_INTERVAL = 1
NUM_TRAIN_ITERATIONS = 10
LEARNING_RATE = 1.
INIT_PARAMETER = 3.
valid_size = 500
train_size = BATCH_SIZE * NUM_UNROLLINGS
vocabulary, train_text, valid_text, _ = load_text_dataset('text8.txt', train_size, valid_size, None)
vocabulary_size = len(vocabulary)
print(vocabulary_size)

env = Environment(
    pupil_class=Lstm,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary)

add_metrics = ['bpc', 'perplexity', 'accuracy']
train_add_feed = [
    {'placeholder': 'dropout', 'value': 1.}
]
valid_add_feed = [
    {'placeholder': 'dropout', 'value': 1.}
]

dataset_name = 'valid'

tf.set_random_seed(1)

env.build_pupil(
    batch_size=BATCH_SIZE,
    num_layers=1,
    num_nodes=[100],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=150,
    num_unrollings=NUM_UNROLLINGS,
    num_gpus=1,
    init_parameter=INIT_PARAMETER,
    regime='autonomous_training',
    additional_metrics=add_metrics,
    going_to_limit_memory=True,
    optimizer='sgd'
)
env.train(
    allow_growth=True,
    restore_path=os.path.join(*(['..']*ROOT_HEIGHT + ['lstm', 'start', 'checkpoints', 'start'])),
    save_path='after_1_step',
    result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
    additions_to_feed_dict=train_add_feed,
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    # stop=stop_specs,
    stop=0,
    vocabulary=vocabulary,
    num_unrollings=NUM_UNROLLINGS,
    results_collect_interval=RESULTS_COLLECT_INTERVAL,
    learning_rate=dict(
        type='exponential_decay',
        decay=1.,
        init=LEARNING_RATE,
        period=1e+6,
    ),
    # opt_inf_results_collect_interval=1,
    summary=False,
    add_graph_to_summary=False,
    train_dataset_text=train_text,
    validation_datasets=dict(valid=valid_text),
    batch_size=BATCH_SIZE
)
env.train(
    allow_growth=True,
    restore_path='after_1_step/checkpoints/final',
    result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
    additions_to_feed_dict=train_add_feed,
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    # stop=stop_specs,
    stop=1,
    vocabulary=vocabulary,
    num_unrollings=NUM_UNROLLINGS,
    results_collect_interval=RESULTS_COLLECT_INTERVAL,
    learning_rate=dict(
        type='exponential_decay',
        decay=1.,
        init=LEARNING_RATE,
        period=1e+6,
    ),
    # opt_inf_results_collect_interval=1,
    summary=False,
    add_graph_to_summary=False,
    train_dataset_text=train_text,
    validation_datasets=dict(valid=valid_text),
    batch_size=BATCH_SIZE
)
