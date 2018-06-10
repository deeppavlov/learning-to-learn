import tensorflow as tf

ROOT_HEIGHT = 6
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
from learning_to_learn.useful_functions import create_vocabulary, compose_hp_confs
from learning_to_learn.optimizers.chiterm import ChiTerm

import os

chi_application = sys.argv[1]
parameter_set_file_name = sys.argv[2]
if len(sys.argv) > 3:
    chop_last_experiment = bool(sys.argv[3])
else:
    chop_last_experiment = False
save_path = os.path.join(parameter_set_file_name.split('.')[0], 'evaluation')
confs, _ = compose_hp_confs(
    parameter_set_file_name,
    os.path.join(save_path, 'valid.txt'),
    chop_last_experiment=chop_last_experiment,
    model='pupil'
)
confs.reverse()  # start with small configs
print("confs:", confs)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
dataset_path = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'text8.txt']))
with open(dataset_path, 'r') as f:
    text = f.read()

valid_size = 10000
valid_text = text[:valid_size]
train_text = text[valid_size:]

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(
    pupil_class=Lstm,
    meta_optimizer_class=ChiTerm,
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
evaluation = dict(
    save_path=save_path,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    datasets=[(valid_text, dataset_name)],
    batch_gen_class=BatchGenerator,
    batch_kwargs={'vocabulary': vocabulary},
    batch_size=1,
    additional_feed_dict=[{'placeholder': 'dropout', 'value': 1.}]
)

BATCH_SIZE = 32
NUM_UNROLLINGS = 10
kwargs_for_building = dict(
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
    going_to_limit_memory=True,
    additional_metrics=add_metrics,
)

meta_optimizer_build_kwargs = dict(
    regime='inference',
    additional_metrics=add_metrics,
    chi_application=chi_application,
)

# stop_specs = dict(
#     type='while_progress',
#     max_no_progress_points=10,
#     changing_parameter_name='learning_rate',
#     path_to_target_metric_storage=('valid', 'loss')
# )
add_feed = [
    {'placeholder': 'dropout', 'value': .9},
    dict(
        placeholder='learning_rate',
        value=4.
    ),
]
launch_kwargs = dict(
    # gpu_memory=.3,
    num_unrollings=NUM_UNROLLINGS,
    vocabulary=vocabulary,
    with_meta_optimizer=True,
    allow_growth=True,
    # save_path=save_path,
    # restore_path='lstm_sample_test/scipop3_1000_bs256_11.12/checkpoints/2000',
    batch_size=BATCH_SIZE,
    checkpoint_steps=None,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=1000,
    # stop=4000,
    train_dataset_text=train_text,
    # train_dataset_text='abc',
    validation_dataset_texts=[valid_text],
    results_collect_interval=100,
    additions_to_feed_dict=add_feed,
    validation_additions_to_feed_dict=valid_add_feed,
    no_validation=True,
)

tf.set_random_seed(1)
for conf in confs:
    build_hyperparameters = dict(
    )
    # other_hyperparameters={'dropout': [.3, .5, .7, .8, .9, .95]},
    other_hyperparameters = dict(
        learning_rate=dict(
            varying=dict(
                init=conf['learning_rate/init']
            ),
            fixed=dict(
                decay=1.,
                period=1e+6
            ),
            hp_type='built-in',
            type='exponential_decay'
        ),
        chi_contribution=dict(
            hp_type='additional_placeholder',
            varying=conf['chi_contribution/value'],
        )
    )

    env.grid_search(
        evaluation,
        kwargs_for_building,
        build_hyperparameters=build_hyperparameters,
        other_hyperparameters=other_hyperparameters,
        meta_optimizer_build_kwargs=meta_optimizer_build_kwargs,
        **launch_kwargs
    )
