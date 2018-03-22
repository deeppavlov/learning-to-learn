import re
from environment import Environment
# from gru_par import Gru, BatchGenerator
from lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from res_net_opt import ResNet4Lstm
from useful_functions import create_vocabulary, get_positions_in_vocabulary

with open('datasets/scipop_v3.0/scipop_train.txt', 'r', encoding='utf-8') as f:
    train_text = re.sub('<[^>]*>', '', f.read( ))

with open('datasets/scipop_v3.0/scipop_valid.txt', 'r', encoding='utf-8') as f:
    valid_text = re.sub('<[^>]*>', '', ''.join(f.readlines()[:10]))

vocabulary = create_vocabulary(train_text + valid_text)
vocabulary_size = len(vocabulary)

env = Environment(
    pupil_class=Lstm,
    meta_optimizer_class=ResNet4Lstm,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary)

env.build_pupil(
    batch_size=64,
    num_layers=2,
    num_nodes=[400, 400],
    num_output_layers=2,
    num_output_nodes=[650],
    vocabulary_size=vocabulary_size,
    embedding_size=150,
    num_unrollings=50,
    init_parameter=3.,
    num_gpus=1)

env.build_optimizer()


add_feed = [{'placeholder': 'dropout', 'value': 0.9}]
valid_add_feed = [{'placeholder': 'dropout', 'value': 1.}]

env.train(
    with_meta_optimizer=True,
    save_path='lstm_sample_test/scipop3_1000_bs256_11.12',
    batch_size=64,
    num_unrollings=50,
    vocabulary=vocabulary,
    checkpoint_steps=2000,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=40000,
    train_dataset_text=train_text,
    validation_dataset_texts=[valid_text],
    results_collect_interval=100,
    additions_to_feed_dict=add_feed,
    validation_additions_to_feed_dict=valid_add_feed)
