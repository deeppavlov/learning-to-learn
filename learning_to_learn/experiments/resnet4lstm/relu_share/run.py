import tensorflow as tf

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
from learning_to_learn.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from learning_to_learn.useful_functions import create_vocabulary, get_hps

from learning_to_learn.optimizers.res_net_opt import ResNet4Lstm

import os

pretrain_step = sys.argv[1]
parameter_set_file_name = sys.argv[2]
if len(sys.argv) > 3:
    initial_experiment_counter_value = int(sys.argv[3])
else:
    initial_experiment_counter_value = 0
hps = get_hps(parameter_set_file_name)
save_path = os.path.join(parameter_set_file_name.split('.')[0], 'evaluation')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
dataset_path = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'text8.txt']))
with open(dataset_path, 'r') as f:
    text = f.read()

valid_size = 500
valid_text = text[:valid_size]
train_text = text[valid_size:]

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(
    pupil_class=Lstm,
    meta_optimizer_class=ResNet4Lstm,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary)

add_metrics = ['bpc', 'perplexity', 'accuracy']
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

NUM_EXERCISES = 10
NUM_UNROLLINGS = 4
BATCH_SIZE = 32
SHARE_TRAIN_DATA = True
checkpoints_path = os.path.join(*(['..']*ROOT_HEIGHT + ['lstm', 'text8_pretrain', 'checkpoints']))
the_only_pupil_restore_path = os.path.join(checkpoints_path, '%s') % pretrain_step
evaluation = dict(
    save_path=save_path,
    opt_inf_is_performed=True,
    opt_inf_stop=20,
    opt_inf_pupil_restore_paths={
        ('pretrain%s' % pretrain_step, the_only_pupil_restore_path)
    },
    opt_inf_additions_to_feed_dict=opt_inf_add_feed,
    opt_inf_validation_dataset_texts=[valid_text],
    opt_inf_train_dataset_texts=[train_text],
    opt_inf_results_collect_interval=1,
    validation_additions_to_feed_dict=valid_add_feed
)

kwargs_for_pupil_building = dict(
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
    regime='training_with_meta_optimizer',
    additional_metrics=add_metrics,
    going_to_limit_memory=True
)

kwargs_for_optimizer_building = dict(
    regime='train',
    # regime='inference',
    num_optimizer_unrollings=10,
    num_exercises=NUM_EXERCISES,
    res_size=2000,
    permute=False,
    optimizer_for_opt_type='adam',
    additional_metrics=add_metrics
)

build_pupil_hyperparameters = dict(
)
build_optimizer_hyperparameters = dict(
    optimizer_init_parameter=hps['optimizer_init_parameter'],
    clip_norm=hps['clip_norm']
)

# other_hyperparameters={'dropout': [.3, .5, .7, .8, .9, .95]},
other_hyperparameters = dict(
    learning_rate=dict(
        varying=dict(
            init=hps['learning_rate']
        ),
        fixed=dict(
            decay=.1,
            period=1e+4
        ),
        hp_type='built-in',
        type='exponential_decay'
    )
)

launch_kwargs = dict(
    allow_growth=True,
    result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
    additions_to_feed_dict=train_opt_add_feed,
    pupil_restore_paths=[the_only_pupil_restore_path],
    reset_period=1,
    num_exercises=NUM_EXERCISES,
    stop=1000,
    train_dataset_texts=[train_text],
    share_train_data=SHARE_TRAIN_DATA,
    opt_inf_is_performed=False,
    vocabulary=vocabulary,
    batch_size=BATCH_SIZE,
    num_unrollings=NUM_UNROLLINGS,
    results_collect_interval=200,
    permute=False,
    summary=True,
    add_graph_to_summary=True
)

tf.set_random_seed(1)
env.grid_search_for_meta(
    evaluation,
    kwargs_for_pupil_building,
    kwargs_for_optimizer_building,
    build_pupil_hyperparameters=build_pupil_hyperparameters,
    build_optimizer_hyperparameters=build_optimizer_hyperparameters,
    other_hyperparameters=other_hyperparameters,
    initial_experiment_counter_value=initial_experiment_counter_value,
    **launch_kwargs
)