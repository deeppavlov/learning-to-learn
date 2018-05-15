import re
import tensorflow as tf

from learning_to_learn.environment import Environment
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary

from learning_to_learn.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator

f = open('datasets/scipop_v3.0/scipop_train.txt', 'r', encoding='utf-8')
train_text = re.sub('<[^>]*>', '', f.read( ))
f.close()

f = open('datasets/scipop_v3.0/scipop_valid.txt', 'r', encoding='utf-8')
valid_text = re.sub('<[^>]*>', '', ''.join(f.readlines()[:10]))
f.close()


vocabulary = create_vocabulary(train_text + valid_text)
vocabulary_size = len(vocabulary)

env = Environment(Lstm, BatchGenerator, vocabulary=vocabulary)

# env = Environment(Gru, BatchGenerator)
cpiv = get_positions_in_vocabulary(vocabulary)

connection_interval = 8
connection_visibility = 5
subsequence_length_in_intervals = 10


add_feed = [{'placeholder': 'dropout', 'value': 0.9} #,
            # {'placeholder': 'sampling_prob',
            #  'value': {'type': 'linear', 'start': 0., 'end': 1., 'interval': 3000}},
            # {'placeholder': 'loss_comp_prob',
            #  'value': {'type': 'linear', 'start': 1., 'end': 0., 'interval': 3000}}
            ]
valid_add_feed = [# {'placeholder': 'sampling_prob', 'value': 1.},
                  {'placeholder': 'dropout', 'value': 1.}]

add_metrics = ['bpc', 'perplexity', 'accuracy']

tf.set_random_seed(1)

env.build_pupil(
    batch_size=32,
    num_layers=1,
    num_nodes=[100],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=150,
    num_unrollings=10,
    init_parameter=3.,
    # character_positions_in_vocabulary=cpiv,
    num_gpus=1,
    additional_metrics=add_metrics,
    going_to_limit_memory=True
)

print('building is finished')
env.train(
    # gpu_memory=.3,
    allow_growth=True,
    save_path='lstm/test_res_net_1000_emb150_nl1_nn100_bs32_nu10',
    # restore_path='lstm_sample_test/scipop3_1000_bs256_11.12/checkpoints/2000',
    learning_rate={'type': 'exponential_decay',
                   'init': 4.,
                   'decay': .1,
                   'period': 13000},
    batch_size=32,
    num_unrollings=10,
    vocabulary=vocabulary,
    checkpoint_steps=200,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=40000,
    train_dataset_text=train_text,
    # train_dataset_text='abc',
    validation_dataset_texts=[valid_text],
    results_collect_interval=100,
    additions_to_feed_dict=add_feed,
    validation_additions_to_feed_dict=valid_add_feed
)
          #log_device_placement=True)
