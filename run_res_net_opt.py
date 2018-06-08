import re

from learning_to_learn.environment import Environment
from learning_to_learn.optimizers.res_net_opt import ResNet4Lstm
from learning_to_learn.pupils.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from learning_to_learn.useful_functions import create_vocabulary

with open('datasets/scipop_v3.0/scipop_train.txt', 'r') as f:
    train_text = re.sub('<[^>]*>', '', f.read())

with open('datasets/scipop_v3.0/scipop_valid.txt', 'r') as f:
    valid_text = re.sub('<[^>]*>', '', ''.join(f.readlines()[:10]))

vocabulary = create_vocabulary(train_text + valid_text)
vocabulary_size = len(vocabulary)

env = Environment(
    pupil_class=Lstm,
    meta_optimizer_class=ResNet4Lstm,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary)

add_metrics = ['bpc', 'perplexity', 'accuracy']

env.build_pupil(
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

env.build_optimizer(
    regime='train',
    # regime='inference',
    num_optimizer_unrollings=10,
    num_exercises=5,
    res_size=2000,
    permute=False,
    optimizer_for_opt_type='adam',
    additional_metrics=add_metrics
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
#     save_path='debug_empty_meta_optimizer/not_learning_issue_es20_nn20',
#     batch_size=32,
#     num_unrollings=3,
#     vocabulary=vocabulary,
#     checkpoint_steps=2000,
#     result_types=['loss'],
#     printed_result_types=['loss'],
#     stop=40000,
#     train_dataset_text=train_text,
#     validation_dataset_texts=[valid_text],
#     results_collect_interval=1,
#     additions_to_feed_dict=opt_inf_add_feed,
#     validation_additions_to_feed_dict=valid_add_feed,
#     summary=True,
#     add_graph_to_summary=True
# ) 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18,
for idx in [50, 0, 1, 10, 20, 24, 28, 32, 36, 40, 50, 60, 80, 100, 120, 160, 200]:
    step = idx * 200
    env.train_optimizer(
        allow_growth=True,
        save_path='res_net_relu/from_%s' % step,
        result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
        additions_to_feed_dict=train_opt_add_feed,
        pupil_restore_paths=['lstm/test_res_net_1000_emb150_nl1_nn100_bs32_nu10/checkpoints/%s' % step],
        # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
        reset_period=1,
        stop=41,
        train_dataset_texts=[train_text],
        opt_inf_is_performed=True,
        opt_inf_stop=10,
        opt_inf_pupil_restore_paths=[
            ('prelearn%s' % step, 'lstm/test_res_net_1000_emb150_nl1_nn100_bs32_nu10/checkpoints/%s' % step)
        ],
        opt_inf_additions_to_feed_dict=opt_inf_add_feed,
        opt_inf_validation_dataset_texts=[valid_text],
        opt_inf_train_dataset_texts=[train_text],
        validation_additions_to_feed_dict=valid_add_feed,
        vocabulary=vocabulary,
        batch_size=32,
        batch_gen_init_is_random=False,
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
