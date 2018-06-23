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
from learning_to_learn.pupils.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from learning_to_learn.useful_functions import create_vocabulary, compose_hp_confs, get_num_exps_and_res_files, \
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
dataset_path = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'text8.txt']))
with open(dataset_path, 'r') as f:
    text = f.read()

valid_size = 2000
test_size = 100000
test_text = text[:test_size]
valid_text = text[test_size:valid_size+test_size]
train_text = text[valid_size+test_size:]

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(
    pupil_class=Lstm,
    meta_optimizer_class=L2L,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary
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

NUM_EXERCISES = 1
BATCH_SIZE = 32
NUM_UNROLLINGS = 10
NUM_OPTIMIZER_UNROLLINGS = 10
RESET_PERIOD = 1
OPT_INF_STOP = NUM_OPTIMIZER_UNROLLINGS * RESET_PERIOD
OPTIMIZER_RANGE = NUM_OPTIMIZER_UNROLLINGS * RESET_PERIOD * NUM_UNROLLINGS
AVERAGING_NUMBER = 3
NUM_OPTIMIZER_TRAIN_STEPS = 1000
OPTIMIZER_TEST_RANGE = 500
OPT_INF_NAME = 'COLD'
OPT_INF_RESTORE_PUPIL_PATHS = [
    (OPT_INF_NAME, None)
]
LSTM_SIZE = dict(
    num_layers=1,
    num_nodes=[100],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=150,
    num_unrollings=NUM_UNROLLINGS,
)
OPTIMIZER_PARAMETERS = dict(
    regime='train',
    # regime='inference',
    num_optimizer_unrollings=NUM_OPTIMIZER_UNROLLINGS,
    num_exercises=NUM_EXERCISES,
    num_lstm_layers=2,
    num_lstm_nodes=[20, 20],
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
    opt_inf_validation_dataset_texts=[valid_text],
    opt_inf_train_dataset_texts=[train_text],
    opt_inf_results_collect_interval=1,
    validation_additions_to_feed_dict=valid_add_feed
)

kwargs_for_pupil_building = dict(
    batch_size=BATCH_SIZE,
    **LSTM_SIZE,
    regime='training_with_meta_optimizer',
    additional_metrics=add_metrics,
    going_to_limit_memory=True
)

kwargs_for_optimizer_building = dict(
    **OPTIMIZER_PARAMETERS,
)

launch_kwargs = dict(
        allow_growth=True,
        # save_path='debug_grid_search',
        result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
        additions_to_feed_dict=train_opt_add_feed,
        # pupil_restore_paths=PUPIL_RESTORE_PATHS,
        # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
        reset_period=RESET_PERIOD,
        stop=NUM_OPTIMIZER_TRAIN_STEPS,
        train_dataset_texts=[train_text],
        opt_inf_is_performed=False,
        num_exercises=NUM_EXERCISES,
        vocabulary=vocabulary,
        batch_size=BATCH_SIZE,
        num_unrollings=NUM_UNROLLINGS,
        results_collect_interval=200,
        # opt_inf_results_collect_interval=1,
        summary=False,
        add_graph_to_summary=False
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
                period=200
            ),
            hp_type='built-in',
            type='exponential_decay'
        )
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

metric_res = best[OPT_INF_NAME]['loss']

best_on_valid = metric_res['validation']
print(' ' * 2 + 'loss' + ':', best_on_valid[1])
print_hps(hp_names, best_on_valid[0], 4)
best_conf = dict(list(zip(hp_names, best_on_valid[0])))
env.build_pupil(
    batch_size=BATCH_SIZE,
    **LSTM_SIZE,
    regime='training_with_meta_optimizer',
    additional_metrics=add_metrics,
    going_to_limit_memory=True,
)

env.build_optimizer(
    **OPTIMIZER_PARAMETERS,
    optimizer_init_parameter=best_conf['optimizer_init_parameter'],
)


stop_specs = 20000

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
    # pupil_restore_paths=[the_only_pupil_restore_path],
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    reset_period=RESET_PERIOD,
    num_exercises=NUM_EXERCISES,
    stop=stop_specs,
    train_dataset_texts=[train_text],
    opt_inf_is_performed=True,
    opt_inf_stop=OPT_INF_STOP,
    opt_inf_pupil_restore_paths=OPT_INF_RESTORE_PUPIL_PATHS,
    opt_inf_additions_to_feed_dict=opt_inf_add_feed,
    opt_inf_validation_dataset_texts=[valid_text],
    opt_inf_train_dataset_texts=[train_text],
    validation_additions_to_feed_dict=valid_add_feed,
    vocabulary=vocabulary,
    batch_size=BATCH_SIZE,
    batch_gen_init_is_random=True,
    num_unrollings=NUM_UNROLLINGS,
    learning_rate=learning_rate,
    results_collect_interval=100,
    opt_inf_results_collect_interval=1,
    permute=False,
    summary=True,
    add_graph_to_summary=True
)

env.train(
    # gpu_memory=.3,
    num_unrollings=NUM_UNROLLINGS,
    vocabulary=vocabulary,
    with_meta_optimizer=True,
    # restore_path=the_only_pupil_restore_path,
    restore_optimizer_path=os.path.join(training_path, 'checkpoints', 'final'),
    save_path=os.path.join(base, 'loss_best', 'test', 'pupil_training'),
    allow_growth=True,
    batch_size=BATCH_SIZE,
    checkpoint_steps=None,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=OPTIMIZER_TEST_RANGE,
    # stop=4000,
    train_dataset_text=train_text,
    validation_dataset_texts=[valid_text],
    results_collect_interval=10,
    additions_to_feed_dict=opt_inf_add_feed,
    validation_additions_to_feed_dict=valid_add_feed,
    no_validation=False,
)

env.test(
    restore_path=os.path.join(base, 'loss_best', 'test', 'pupil_training', 'checkpoints/final'),
    save_path=os.path.join(base, 'loss_best', 'test', 'testing'),
    additions_to_feed_dict=valid_add_feed,
    validation_dataset_texts=[test_text],
    valid_batch_kwargs=dict(
        vocabulary=vocabulary
    ),
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy']
)
