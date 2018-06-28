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
    optimizer_time_measurement_save_order, save_lines, extend_for_relative

from learning_to_learn.optimizers.ff import Ff

import os

conf_file = sys.argv[1]
save_path = os.path.join(conf_file.split('.')[0], 'results')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

with open(conf_file, 'r') as f:
    lines = f.read().split('\n')
model = lines[0]
steps = int(lines[1])
base = lines[2]
if base == 'None':
    base = None
else:
    base = float(base)
names = lines[3].split()
types = lines[4].split()
optimizer_varying = dict()
for name, type_, line in zip(names, types, lines[5:]):
    optimizer_varying[name] = [convert(v, type_) for v in line.split()]

dataset_path = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'text8.txt']))
with open(dataset_path, 'r') as f:
    text = f.read()

train_text = text

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)
print(vocabulary_size)

env = Environment(
    pupil_class=Lstm,
    meta_optimizer_class=Ff,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary,
)

add_metrics = ['bpc', 'perplexity', 'accuracy']
NUM_EXERCISES = 10
NUM_UNROLLINGS = 10
OPT_INF_RESTORE_PUPIL_PATHS = [
    ('COLD', None)
]
PUPIL_RESTORE_PATHS = [
    None
]

BATCH_SIZE = 32
pupil_build = dict(
    batch_size=BATCH_SIZE,
    num_layers=1,
    num_nodes=[100],
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

optimizer_launch = dict(
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

pupil_launch = dict(
    # gpu_memory=.3,
    num_unrollings=NUM_UNROLLINGS,
    vocabulary=vocabulary,
    with_meta_optimizer=True,
    # restore_path=the_only_pupil_restore_path,
    allow_growth=True,
    batch_size=BATCH_SIZE,
    checkpoint_steps=None,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=steps,
    # stop=4000,
    train_dataset_text=train_text,
    results_collect_interval=1000,
    additions_to_feed_dict=opt_inf_add_feed,
    validation_additions_to_feed_dict=valid_add_feed,
    no_validation=True,
)

if model == 'optimizer':
    launch = optimizer_launch
elif model == 'pupil':
    launch = pupil_launch
else:
    launch = None

times = env.iter_time(
    steps,
    base,
    pupil_build,
    optimizer_build,
    launch,
    dict(),
    optimizer_varying,
    dict(),
    model=model,
)
print(times)
times = extend_for_relative(times)
print(times)
order = optimizer_time_measurement_save_order(names, base)
print(order)
times = transform_data_into_dictionary_of_lines(times, order)
print(times)
save_lines(times, save_path)
