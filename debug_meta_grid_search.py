import re

from learning_to_learn.environment import Environment
from learning_to_learn.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from learning_to_learn.optimizers.res_net_opt import ResNet4Lstm
from learning_to_learn.useful_functions import create_vocabulary

with open('datasets/text8.txt', 'r') as f:
    text = re.sub('<[^>]*>', '', f.read( ))

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)
# print("(debug_meta_grid_search)vocabulary_size:", vocabulary_size)

valid_size = 500
valid_text = text[:valid_size]
train_text = text[valid_size:]

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

NUM_EXERCISES = 11
evaluation = dict(
    save_path='debug_grid_search/15ex/evaluation',
    opt_inf_is_performed=True,
    opt_inf_stop=10,
    opt_inf_pupil_restore_paths={
        ('prelearn4000', 'lstm/text8_pretrain/checkpoints/200')
    },
    opt_inf_additions_to_feed_dict=opt_inf_add_feed,
    opt_inf_validation_dataset_texts=[valid_text],
    opt_inf_train_dataset_texts=[train_text],
    validation_additions_to_feed_dict=valid_add_feed
)

kwargs_for_pupil_building = dict(
    batch_size=32,
    num_layers=1,
    num_nodes=[100],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=150,
    num_unrollings=4,
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
    init_parameter=[.5, 1., 1.5, 2., 2.7, 4.]
)
build_optimizer_hyperparameters = dict(
    num_optimizer_unrollings=[10, 7, 5]
)

# other_hyperparameters={'dropout': [.3, .5, .7, .8, .9, .95]},
other_hyperparameters = dict(
    dropout=[.3, .5, .7, .8, .9, .95]
)

launch_kwargs = dict(
    allow_growth=True,
    # save_path='debug_grid_search',
    result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
    additions_to_feed_dict=train_opt_add_feed,
    pupil_restore_paths=['lstm/text8_pretrain/checkpoints/200'],
    # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
    reset_period=1,
    stop=41,
    train_dataset_texts=[train_text],
    opt_inf_is_performed=False,
    num_exercises=NUM_EXERCISES,
    # opt_inf_stop=10,
    # opt_inf_pupil_restore_paths={
    #     'prelearn2000': 'lstm/test_res_net_1000_emb150_nl1_nn100_bs32_nu10/checkpoints/2000'
    # },
    # opt_inf_additions_to_feed_dict=opt_inf_add_feed,
    # opt_inf_validation_dataset_texts=[valid_text],
    # opt_inf_train_dataset_texts=[train_text],
    # validation_additions_to_feed_dict=valid_add_feed,
    vocabulary=vocabulary,
    batch_size=32,
    num_unrollings=4,
    learning_rate={'type': 'exponential_decay',
                   'init': .002,
                   'decay': .5,
                   'period': 400},
    results_collect_interval=10,
    opt_inf_results_collect_interval=1,
    permute=False,
    summary=True,
    add_graph_to_summary=True
)

env.grid_search_for_meta(
    evaluation,
    kwargs_for_pupil_building,
    kwargs_for_optimizer_building,
    build_pupil_hyperparameters=build_pupil_hyperparameters,
    build_optimizer_hyperparameters=build_optimizer_hyperparameters,
    other_hyperparameters=other_hyperparameters,
    **launch_kwargs
)
