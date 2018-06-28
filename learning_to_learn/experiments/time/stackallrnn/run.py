ROOT_HEIGHT = 4
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[ROOT_HEIGHT]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass

from learning_to_learn.environment import Environment
from learning_to_learn.pupils.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from learning_to_learn.useful_functions import create_vocabulary, convert, transform_data_into_dictionary_of_lines, \
    optimizer_time_measurement_save_order, save_lines, extend_for_relative, create_path

from learning_to_learn.optimizers.stackallrnn import StackAllRnn

import os

conf_file = sys.argv[1]
save_path = os.path.join(conf_file.split('.')[0], 'results')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

with open(conf_file, 'r') as f:
    lines = f.read().split('\n')
steps = int(lines[0])
base = lines[1]
if base == 'None':
    base = None
else:
    base = float(base)

dataset_path = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'text8.txt']))
with open(dataset_path, 'r') as f:
    text = f.read()

train_text = text

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)
print(vocabulary_size)

env = Environment(
    pupil_class=Lstm,
    meta_optimizer_class=StackAllRnn,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary,
)

add_metrics = ['bpc', 'perplexity', 'accuracy']
NUM_EXERCISES = 1
NUM_UNROLLINGS = 1
OPT_INF_RESTORE_PUPIL_PATHS = [
    ('COLD', None)
]
PUPIL_RESTORE_PATHS = [
    None
]
BATCH_SIZE = 2
pupil_build = dict(
    batch_size=BATCH_SIZE,
    num_layers=1,
    num_nodes=[10],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=150,
    num_unrollings=NUM_UNROLLINGS,
    init_parameter=3.,
    num_gpus=1,
    regime='optimizer_training',
    additional_metrics=add_metrics,
    going_to_limit_memory=True
)

optimizer_build = dict(
    regime='train',
    # regime='inference',
    num_optimizer_unrollings=10,
    num_exercises=NUM_EXERCISES,
    additional_metrics=add_metrics,
    clip_norm=1000000.,
    optimizer_init_parameter=.01
)


train_opt_add_feed = [
    {'placeholder': 'dropout', 'value': .9},
    {'placeholder': 'optimizer_dropout_keep_prob', 'value': .9}
]
opt_inf_add_feed = [
    {'placeholder': 'dropout', 'value': .9},
    {'placeholder': 'optimizer_dropout_keep_prob', 'value': 1.}
]
valid_add_feed = [
    {'placeholder': 'dropout', 'value': 1.},
    {'placeholder': 'optimizer_dropout_keep_prob', 'value': 1.}
]

launch = dict(
    allow_growth=True,
    result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
    additions_to_feed_dict=train_opt_add_feed,
    pupil_restore_paths=PUPIL_RESTORE_PATHS,
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    reset_period=1,
    num_exercises=NUM_EXERCISES,
    train_dataset_texts=[train_text],
    opt_inf_is_performed=False,
    vocabulary=vocabulary,
    batch_size=BATCH_SIZE,
    batch_gen_init_is_random=True,
    num_unrollings=NUM_UNROLLINGS,
    learning_rate={'type': 'exponential_decay',
                   'init': 3e-4,
                   'decay': .1,
                   'period': 3500},
    results_collect_interval=100,
)

times = env.optimizer_iter_time(
    steps,
    base,
    pupil_build,
    optimizer_build,
    launch,
    dict(),
    dict(),
    dict(),
)
times = extend_for_relative(times)
order = optimizer_time_measurement_save_order([], base)
print(order)
print(times)

create_path(save_path, file_name_is_in_path=True)
with open(save_path + '.txt', 'w') as f:
    f.write(str(times))
