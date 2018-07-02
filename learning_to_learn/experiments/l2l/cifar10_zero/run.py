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
from learning_to_learn.pupils.mlp_for_meta import MlpForMeta as Mlp
from learning_to_learn.image_batch_gens import CifarBatchGenerator
from learning_to_learn.useful_functions import compose_hp_confs, get_num_exps_and_res_files, \
    get_optimizer_evaluation_results, get_best, print_hps, get_hp_names_from_conf_file

from learning_to_learn.optimizers.l2l import L2L
import os

parameter_set_file_name = sys.argv[1]


base = parameter_set_file_name.split('.')[0]
save_path = base + '/evaluation'
confs, _ = compose_hp_confs(parameter_set_file_name, save_path, chop_last_experiment=False)
confs.reverse()  # start with small configs
print("confs:", confs)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
VALID_SIZE = 1000

env = Environment(
    pupil_class=Mlp,
    meta_optimizer_class=L2L,
    batch_generator_classes=CifarBatchGenerator,
)

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

# NUM_EXERCISES = 10
# BATCH_SIZE = 32
# NUM_OPTIMIZER_UNROLLINGS = 5
# RESET_PERIOD = 20
# OPT_INF_STOP = RESET_PERIOD * NUM_OPTIMIZER_UNROLLINGS
# RESTORE_PUPIL_PATHS = [
#     the_only_pupil_restore_path
# ]
# OPT_INF_RESTORE_PUPIL_PATHS = [
#     ('adam_prep', the_only_pupil_restore_path)
# ]
# PUPIL_RESTORE_PATHS = [
#     RESTORE_PUPIL_PATHS[0]
# ]
# OPTIMIZER_RANGE = NUM_OPTIMIZER_UNROLLINGS * RESET_PERIOD
# AVERAGING_NUMBER = 3
# NUM_OPTIMIZER_TRAIN_STEPS = 1000
# MLP_SIZE = dict(
#     num_layers=2,
#     num_hidden_nodes=[1000],
#     input_shape=[3072],
#     num_classes=10,
# )
# OPTIMIZER_PARAMETERS = dict(
#     regime='train',
#     # regime='inference',
#     num_optimizer_unrollings=NUM_OPTIMIZER_UNROLLINGS,
#     num_exercises=NUM_EXERCISES,
#     res_size=1000,
#     num_res_layers=4,
#     permute=False,
#     optimizer_for_opt_type='adam',
#     additional_metrics=add_metrics
# )

NUM_EXERCISES = 1
BATCH_SIZE = 32
NUM_OPTIMIZER_UNROLLINGS = 3
RESET_PERIOD = 33
OPT_INF_STOP = NUM_OPTIMIZER_UNROLLINGS * RESET_PERIOD
PUPIL_NAME = 'COLD'
OPT_INF_RESTORE_PUPIL_PATHS = [
    (PUPIL_NAME, None)
]
OPTIMIZER_RANGE = NUM_OPTIMIZER_UNROLLINGS * RESET_PERIOD
AVERAGING_NUMBER = 3
NUM_OPTIMIZER_TRAIN_STEPS = 1000
OPTIMIZER_TEST_RANGE = 500
MLP_SIZE = dict(
    num_layers=2,
    num_hidden_nodes=[1000],
    input_shape=[3072],
    num_classes=10,
    init_parameter=0.1,
)
OPTIMIZER_PARAMETERS = dict(
    regime='train',
    # regime='inference',
    num_optimizer_unrollings=NUM_OPTIMIZER_UNROLLINGS,
    num_exercises=NUM_EXERCISES,
    num_lstm_layers=1,
    num_lstm_nodes=[10],
    selected=['omega', 'beta'],
    optimizer_for_opt_type='adam',
    additional_metrics=add_metrics,
    get_omega_and_beta=True,
)
evaluation = dict(
    save_path=save_path,
    opt_inf_is_performed=True,
    opt_inf_stop=OPT_INF_STOP,
    opt_inf_pupil_restore_paths=OPT_INF_RESTORE_PUPIL_PATHS,
    opt_inf_additions_to_feed_dict=opt_inf_add_feed,
    opt_inf_validation_datasets=[['validation', 'valid']],
    opt_inf_train_datasets=[['train', 'train']],
    opt_inf_results_collect_interval=1,
    validation_additions_to_feed_dict=valid_add_feed
)

kwargs_for_pupil_building = dict(
    batch_size=BATCH_SIZE,
    **MLP_SIZE,
    regime='training_with_meta_optimizer',
    additional_metrics=add_metrics,
)

kwargs_for_optimizer_building = dict(
    **OPTIMIZER_PARAMETERS,
)

launch_kwargs = dict(
    allow_growth=True,
    # save_path='debug_grid_search',
    result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
    additions_to_feed_dict=train_opt_add_feed,
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    reset_period=RESET_PERIOD,
    stop=NUM_OPTIMIZER_TRAIN_STEPS,
    train_datasets=[('train', 'train')],
    opt_inf_is_performed=False,
    num_exercises=NUM_EXERCISES,
    batch_size=BATCH_SIZE,
    results_collect_interval=200,
    # opt_inf_results_collect_interval=1,
    summary=False,
    add_graph_to_summary=False,
    train_batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),
    valid_batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),
    one_batch_gen=True,
)

for conf in confs:
    build_pupil_hyperparameters = dict(
    )
    build_optimizer_hyperparameters = dict(
        optimizer_init_parameter=conf['optimizer_init_parameter'],
    )

    # other_hyperparameters={'dropout': [.3, .5, .7, .8, .9, .95]},
    other_hyperparameters = dict(
        learning_rate=dict(
            varying=dict(
                init=conf['learning_rate/init']
            ),
            fixed=dict(
                decay=.5,
                period=2000
            ),
            hp_type='built-in',
            type='exponential_decay'
        ),
    )


    tf.set_random_seed(1)
    _, biggest_idx, _ = get_num_exps_and_res_files(save_path)
    if biggest_idx is None:
        initial_experiment_counter_value = 0
    else:
        initial_experiment_counter_value = biggest_idx + 1
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


hp_names = get_hp_names_from_conf_file(parameter_set_file_name)
for_plotting = get_optimizer_evaluation_results(save_path, hp_names,  AVERAGING_NUMBER)

best = get_best(for_plotting, 'optimizer')

metric_res = best[PUPIL_NAME]['loss']

best_on_valid = metric_res['validation']
print(' ' * 2 + 'loss' + ':', best_on_valid[1])
print_hps(hp_names, best_on_valid[0], 4)
best_conf = dict(list(zip(hp_names, best_on_valid[0])))
env.build_pupil(
    batch_size=BATCH_SIZE,
    **MLP_SIZE,
    regime='training_with_meta_optimizer',
    additional_metrics=add_metrics,
)

env.build_optimizer(
    **OPTIMIZER_PARAMETERS,
    optimizer_init_parameter=best_conf['optimizer_init_parameter'],
)

stop_specs = 20000        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

learning_rate = dict(
    type='exponential_decay',
    period=4000,
    decay=.5,
    init=best_conf['learning_rate/init'],
)
training_path = os.path.join(base, 'loss_best', 'test', 'training')
env.train_optimizer(
    allow_growth=True,
    save_path=training_path,
    result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
    additions_to_feed_dict=train_opt_add_feed,
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    reset_period=RESET_PERIOD,
    num_exercises=NUM_EXERCISES,
    stop=stop_specs,
    train_datasets=[('train', 'train')],
    opt_inf_is_performed=True,
    opt_inf_stop=OPT_INF_STOP,
    opt_inf_pupil_restore_paths=OPT_INF_RESTORE_PUPIL_PATHS,
    opt_inf_additions_to_feed_dict=opt_inf_add_feed,
    opt_inf_validation_datasets=[['validation', 'valid']],
    opt_inf_train_datasets=[['train', 'train']],
    validation_additions_to_feed_dict=valid_add_feed,
    batch_size=BATCH_SIZE,
    batch_gen_init_is_random=True,
    learning_rate=learning_rate,
    results_collect_interval=2000,
    opt_inf_results_collect_interval=10,
    permute=False,
    summary=True,
    add_graph_to_summary=True,
    one_batch_gen=True,
    train_batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),
    valid_batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),
)

env.train(
    # gpu_memory=.3,
    with_meta_optimizer=True,
    restore_optimizer_path=os.path.join(training_path, 'checkpoints', 'final'),
    save_path=os.path.join(base, 'loss_best', 'test', 'pupil_training'),
    allow_growth=True,
    batch_size=BATCH_SIZE,
    checkpoint_steps=None,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=OPTIMIZER_TEST_RANGE,
    # stop=4000,
    train_dataset=dict(
        train='train'
    ),
    validation_datasets=dict(
        valid='validation'
    ),
    train_batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),
    valid_batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),
    results_collect_interval=1,
    additions_to_feed_dict=opt_inf_add_feed,
    validation_additions_to_feed_dict=valid_add_feed,
    no_validation=False,
)

env.test(
    restore_path=os.path.join(base, 'loss_best', 'test', 'pupil_training', 'checkpoints/final'),
    save_path=os.path.join(base, 'loss_best', 'test', 'testing'),
    additions_to_feed_dict=valid_add_feed,
    validation_datasets=dict(
        test='test'
    ),
    valid_batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy']
)
