import tensorflow as tf

ROOT_HEIGHT = 3
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
from learning_to_learn.useful_functions import create_vocabulary, compose_hp_confs, get_pupil_evaluation_results, \
    print_hps, get_best

import os

parameter_set_file_name = sys.argv[1]
chop_last_experiment = False
base = parameter_set_file_name.split('.')[0]
opt = base
save_path = os.path.join(base, 'evaluation')
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

test_size = 100000
valid_size = 2000
test_text = text[:test_size]
valid_text = text[test_size:test_size+valid_size]
train_text = text[test_size+valid_size:]

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(
    pupil_class=Lstm,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary)

add_metrics = ['bpc', 'perplexity', 'accuracy']
train_add_feed = [
    {'placeholder': 'dropout', 'value': .9}
]

if base in ['nesterov', 'momentum']:
    train_add_feed.append(
        {'placeholder': 'momentum', 'value': 0.97}
    )
valid_add_feed = [
    {'placeholder': 'dropout', 'value': 1.}
]

dataset_name = 'valid'
RESTORE_PATH = '../text8_max_train/adam/2/checkpoints/best'
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
kwargs_for_building = dict(
    batch_size=BATCH_SIZE,
    num_layers=1,
    num_nodes=[100],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=150,
    num_unrollings=10,
    num_gpus=1,
    regime='autonomous_training',
    additional_metrics=add_metrics,
    going_to_limit_memory=True,
    optimizer=opt
)

# stop_specs = dict(
#     type='while_progress',
#     max_no_progress_points=10,
#     changing_parameter_name='learning_rate',
#     path_to_target_metric_storage=('valid', 'loss')
# )
launch_kwargs = dict(
    allow_growth=True,
    # save_path='debug_grid_search',
    result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
    additions_to_feed_dict=train_add_feed,
    restore_path=RESTORE_PATH,
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    # stop=stop_specs,
    stop=1000,
    vocabulary=vocabulary,
    num_unrollings=10,
    results_collect_interval=500,
    # opt_inf_results_collect_interval=1,
    summary=False,
    add_graph_to_summary=False,
    train_dataset_text=train_text,
    validation_datasets=dict(valid=valid_text),
    batch_size=BATCH_SIZE,
    no_validation=True,
)

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
        )
    )

    env.grid_search(
        evaluation,
        kwargs_for_building,
        build_hyperparameters=build_hyperparameters,
        other_hyperparameters=other_hyperparameters,
        **launch_kwargs
    )

hp_names = list(confs[0].keys())
for_plotting = get_pupil_evaluation_results(save_path, hp_names)

best = get_best(for_plotting, 'pupil')

env.build_pupil(
    batch_size=BATCH_SIZE,
    num_layers=1,
    num_nodes=[100],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=150,
    num_unrollings=10,
    num_gpus=1,
    regime='autonomous_training',
    additional_metrics=add_metrics,
    going_to_limit_memory=True,
    optimizer=opt
)
for dataset_name, dataset_res in best.items():
    print('dataset:', dataset_name)
    for metric, b in dataset_res.items():
        print(' ' * 2 + metric + ':', b[1])
        print_hps(hp_names, b[0], 4)
        best_conf = dict(list(zip(hp_names, b[0])))
        training_path = os.path.join(base, metric + '_best', 'test', 'training')

        env.train(
            allow_growth=True,
            # save_path='debug_grid_search',
            result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
            additions_to_feed_dict=train_add_feed,
            restore_path=RESTORE_PATH,
            save_path=training_path,
            # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
            # stop=stop_specs,
            stop=1000,
            vocabulary=vocabulary,
            num_unrollings=10,
            results_collect_interval=500,
            # opt_inf_results_collect_interval=1,
            summary=False,
            add_graph_to_summary=False,
            train_dataset_text=train_text,
            validation_datasets=dict(valid=valid_text),
            batch_size=BATCH_SIZE,
            no_validation=True,
            learning_rate=dict(
                init=best_conf['learning_rate/init'],
                decay=1.,
                period=1e+6,
                type='exponential_decay',
            ),

        )

        env.test(
            restore_path=os.path.join(training_path, 'checkpoints/final'),
            save_path=os.path.join(base, metric + '_best', 'test', 'testing'),
            additions_to_feed_dict=valid_add_feed,
            validation_datasets=dict(
                test='test'
            ),
            valid_batch_kwargs=dict(
                vocabulary=vocabulary
            ),
            printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy']
        )

