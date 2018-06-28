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
from learning_to_learn.pupils.mlp_for_meta import MlpForMeta
from learning_to_learn.useful_functions import create_vocabulary, convert, transform_data_into_dictionary_of_lines, \
    optimizer_time_measurement_save_order, save_lines, extend_for_relative
from learning_to_learn.image_batch_gens import MnistBatchGenerator

from learning_to_learn.optimizers.resnet4mlp import ResNet4Mlp

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
names = lines[2].split()
types = lines[3].split()
optimizer_varying = dict()
for name, type_, line in zip(names, types, lines[4:]):
    optimizer_varying[name] = [convert(v, type_) for v in line.split()]

data_dir = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'mnist']))

env = Environment(
    pupil_class=MlpForMeta,
    meta_optimizer_class=ResNet4Mlp,
    batch_generator_classes=MnistBatchGenerator,
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
    num_hidden_nodes=[],
    input_shape=[784],
    num_classes=10,
    init_parameter=3.,
    additional_metrics=add_metrics,
    optimizer='sgd'
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
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    num_exercises=NUM_EXERCISES,
    reset_period=1,
    stop=steps,
    train_datasets=[('train', 'train')],
    opt_inf_is_performed=False,
    validation_additions_to_feed_dict=valid_add_feed,
    batch_size=BATCH_SIZE,
    batch_gen_init_is_random=True,
    results_collect_interval=2000,
    opt_inf_results_collect_interval=10,
    permute=False,
    summary=True,
    add_graph_to_summary=True,
    one_batch_gen=True,
    train_batch_kwargs=dict(
        data_dir=data_dir
    ),
    valid_batch_kwargs=dict(
        data_dir=data_dir
    ),
)

times = env.iter_time(
    steps,
    base,
    pupil_build,
    optimizer_build,
    launch,
    dict(),
    optimizer_varying,
    dict(),
)
times = extend_for_relative(times)
order = optimizer_time_measurement_save_order(names, base)
print(order)
print(times)
times = transform_data_into_dictionary_of_lines(times, order)
save_lines(times, 'results')
