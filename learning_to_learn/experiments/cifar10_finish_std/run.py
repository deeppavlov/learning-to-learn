import tensorflow as tf

ROOT_HEIGHT = 3
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[ROOT_HEIGHT]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.environment import Environment
from learning_to_learn.pupils.mlp_for_meta import MlpForMeta as Mlp
from learning_to_learn.image_batch_gens import CifarBatchGenerator
from learning_to_learn.useful_functions import compose_hp_confs, get_best, get_pupil_evaluation_results, print_hps

import os

opt = sys.argv[1]
parameter_set_file_name = sys.argv[2]
chop_last_experiment = False
base = parameter_set_file_name.split('.')[0]
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
data_dir = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'mnist']))

env = Environment(Mlp, CifarBatchGenerator)

add_metrics = ['bpc', 'perplexity', 'accuracy']
train_add_feed = [
    {'placeholder': 'dropout', 'value': .9}
]
if base in ['nesterov', 'momentum']:
    train_add_feed.append(
        {'placeholder': 'momentum', 'value': 0.98}
    )
valid_add_feed = [
    {'placeholder': 'dropout', 'value': 1.}
]
VALID_SIZE = 1000

dataset_name = 'valid'
RESTORE_PATH = '../cifar10_max_train/adagrad/0/checkpoints/best'
evaluation = dict(
    save_path=save_path,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    datasets=[
        ('validation', 'valid')
    ],
    batch_gen_class=CifarBatchGenerator,
    batch_kwargs=dict(
        valid_size=VALID_SIZE
    ),
    batch_size=None,
    additional_feed_dict=valid_add_feed,
)

BATCH_SIZE = 32
kwargs_for_building = dict(
    batch_size=BATCH_SIZE,
    num_layers=2,
    num_hidden_nodes=[1000],
    input_shape=[3072],
    num_classes=10,
    init_parameter=3.,
    additional_metrics=add_metrics,
    regularization_rate=1e-5,
    optimizer=opt,
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
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    # stop=stop_specs,
    restore_path=RESTORE_PATH,
    stop=1000,
    results_collect_interval=1000,
    summary=False,
    add_graph_to_summary=False,
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

    tf.set_random_seed(1)
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
    num_layers=2,
    num_hidden_nodes=[1000],
    input_shape=[3072],
    num_classes=10,
    additional_metrics=add_metrics,
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
            # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
            # stop=stop_specs,
            save_path=training_path,
            restore_path=RESTORE_PATH,
            stop=1000,
            results_collect_interval=1000,
            summary=False,
            add_graph_to_summary=False,
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
            learning_rate=dict(
                init=best_conf['learning_rate/init'],
                decay=1.,
                period=1e+6,
                type='exponential_decay',
            ),

            batch_size=BATCH_SIZE,
            no_validation=True,
        )

        env.test(
            restore_path=os.path.join(training_path, 'checkpoints/final'),
            save_path=os.path.join(base, metric + '_best', 'test', 'testing'),
            additions_to_feed_dict=valid_add_feed,
            validation_datasets=dict(
                test='test'
            ),
            valid_batch_kwargs=dict(
                valid_size=VALID_SIZE
            ),
            printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy']
        )
