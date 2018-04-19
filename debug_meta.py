import re
from environment import Environment
from lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from res_net_opt import ResNet4Lstm
from useful_functions import create_vocabulary, get_positions_in_vocabulary

with open('datasets/scipop_v3.0/scipop_train.txt', 'r') as f:
    train_text = re.sub('<[^>]*>', '', f.read( ))

with open('datasets/scipop_v3.0/scipop_valid.txt', 'r') as f:
    valid_text = re.sub('<[^>]*>', '', ''.join(f.readlines()[:10]))

vocabulary = create_vocabulary(train_text + valid_text)
vocabulary_size = len(vocabulary)

env = Environment(
    pupil_class=Lstm,
    meta_optimizer_class=ResNet4Lstm,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary)

env.build_pupil(
    batch_size=32,
    num_layers=1,
    num_nodes=[100],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=150,
    num_unrollings=3,
    init_parameter=3.,
    num_gpus=1,
    regime='training_with_meta_optimizer'
)

env.build_optimizer(
    regime='train',
    num_optimizer_unrollings=7,
    num_exercises=5,
    res_size=250,
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

# env.train(
#     with_meta_optimizer=True,
#     save_path='debug_empty_meta_optimizer/not_changing_variables_issue',
#     batch_size=32,
#     num_unrollings=3,
#     vocabulary=vocabulary,
#     checkpoint_steps=2000,
#     result_types=['loss'],
#     printed_result_types=['loss'],
#     stop=40000,
#     train_dataset_text=train_text,
#     validation_dataset_texts=[valid_text],
#     results_collect_interval=100,
#     additions_to_feed_dict=opt_inf_add_feed,
#     validation_additions_to_feed_dict=valid_add_feed,
#     summary=True,
#     add_graph_to_summary=True
# )

env.train_optimizer(
    save_path='meta_optimizer_training_debug',
    additions_to_feed_dict=train_opt_add_feed,
    pupil_restore_paths=['debug_empty_meta_optimizer/not_changing_variables_issue/checkpoints/0'],
    reset_period=10,
    train_dataset_texts=[train_text],
    opt_inf_is_performed=True,
    opt_inf_stop=100,
    opt_inf_pupil_restore_paths={'ignoramus': 'debug_empty_meta_optimizer/not_changing_variables_issue/checkpoints/0'},
    opt_inf_additions_to_feed_dict=opt_inf_add_feed,
    opt_inf_validation_dataset_texts=[valid_text],
    opt_inf_train_dataset_texts=[train_text],
    validation_additions_to_feed_dict=valid_add_feed,
    vocabulary=vocabulary,
    batch_size=32,
    num_unrollings=3,
    learning_rate={'type': 'exponential_decay',
                   'init': .001,
                   'decay': .5,
                   'period': 400},
    results_collect_interval=30,
    opt_inf_results_collect_interval=1
)
