ROOT_HEIGHT = 3
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

import os

conf_file = sys.argv[1]
save_path = '.'.join(conf_file.split('.')[:-1])
with open(conf_file, 'r') as f:
    lines = f.read().split('\n')
optimizer = lines[0]
freq = int(lines[1])
stop = int(lines[2])
ip = float(lines[3])
lr = float(lines[4])

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

with open(conf_file, 'r') as f:
    lines = f.read().split('\n')

dataset_path = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'text8.txt']))
with open(dataset_path, 'r') as f:
    text = f.read()


valid_size = 2000
test_size = 100000

valid_text = text[test_size:test_size+valid_size]
train_text = text[test_size+valid_size:]

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)
print(vocabulary_size)

env = Environment(
    pupil_class=Lstm,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary)

add_metrics = ['bpc', 'perplexity', 'accuracy']
train_add_feed = [
    {'placeholder': 'dropout', 'value': .9}
]
valid_add_feed = [
    {'placeholder': 'dropout', 'value': 1.}
]

dataset_name = 'valid'

tf.set_random_seed(1)
BATCH_SIZE = 32
NUM_UNROLLINGS = 10
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
    init_parameter=ip,
    regime='autonomous_training',
    additional_metrics=add_metrics,
    going_to_limit_memory=True,
    optimizer=optimizer,
)

tf.set_random_seed(1)
env.train(
    allow_growth=True,
    save_path=save_path,
    result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
    additions_to_feed_dict=train_add_feed,
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    stop=stop,
    vocabulary=vocabulary,
    num_unrollings=NUM_UNROLLINGS,
    results_collect_interval=freq,
    learning_rate=dict(
        type='adaptive_change',
        max_no_progress_points=20,
        decay=.5,
        init=lr,
        path_to_target_metric_storage=('valid', 'loss')
    ),
    # opt_inf_results_collect_interval=1,
    summary=False,
    add_graph_to_summary=False,
    train_dataset_text=train_text,
    validation_datasets=dict(valid=valid_text),
    batch_size=BATCH_SIZE
)
