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
from learning_to_learn.pupils.mlp_for_meta import MlpForMeta as Mlp
from learning_to_learn.image_batch_gens import CifarBatchGenerator
from learning_to_learn.useful_functions import compose_hp_confs
from learning_to_learn.optimizers.chinoise import ChiNoise

import os

fix_seed = sys.argv[1]
chi_application = sys.argv[2]
parameter_set_file_name = sys.argv[3]
if fix_seed == 'True':
    fix_seed = True
elif fix_seed == 'False':
    fix_seed = False
else:
    fix_seed = None
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

VALID_SIZE = 1000

env = Environment(
    pupil_class=Mlp,
    meta_optimizer_class=ChiNoise,
    batch_generator_classes=CifarBatchGenerator,
)

add_metrics = ['bpc', 'perplexity', 'accuracy']

valid_add_feed = [
    {'placeholder': 'dropout', 'value': 1.}
]

dataset_name = 'valid'
evaluation = dict(
    save_path=save_path,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    datasets=[('validation', 'valid')],
    batch_gen_class=CifarBatchGenerator,
    batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),
    batch_size=1,
    additional_feed_dict=[{'placeholder': 'dropout', 'value': 1.}]
)

BATCH_SIZE = 32
kwargs_for_building = dict(
    batch_size=BATCH_SIZE,
    num_layers=2,
    num_hidden_nodes=[1000],
    input_shape=[3072],
    num_classes=10,
    init_parameter=.1,
    additional_metrics=add_metrics,
    regime='training_with_meta_optimizer',
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
]
launch_kwargs = dict(
    # gpu_memory=.3,
    allow_growth=True,
    save_path='debug_early_stop',
    with_meta_optimizer=True,
    # restore_path='lstm_sample_test/scipop3_1000_bs256_11.12/checkpoints/2000',
    batch_size=BATCH_SIZE,
    checkpoint_steps=None,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=1000,
    train_dataset=dict(
        train='train'
    ),
    train_batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),
    valid_batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),

    # train_dataset_text='abc',
    validation_datasets=dict(
        valid='validation'
    ),
    results_collect_interval=100,
    additions_to_feed_dict=add_feed,
    validation_additions_to_feed_dict=valid_add_feed,
    no_validation=False
)

if fix_seed:
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
            hp_type='additional_placeholder',
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
