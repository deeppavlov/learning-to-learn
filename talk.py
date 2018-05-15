
from environment import Environment
from lstm_par import Lstm, LstmBatchGenerator
from learning_to_learn.useful_functions import create_vocabulary, get_positions_in_vocabulary

f = open('datasets/ted.txt', 'r', encoding='utf-8')
text = f.read()
f.close()


# In[5]:

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)

env = Environment(Lstm, LstmBatchGenerator)
cpiv = get_positions_in_vocabulary(vocabulary)

env.build(batch_size=64,
          num_layers=2,
          num_nodes=[1700, 1700],
          num_output_layers=2,
          num_output_nodes=[1024],
          vocabulary_size=vocabulary_size,
          embedding_size=128,
          num_unrollings=1,
          num_gpus=2)

env.inference(restore_path='lstm/big_network_1700_ted_correct_8.11/checkpoints/final',
              log_path='lstm/big_network_1700_ted_correct_8.11/dialogs/1',
              vocabulary=vocabulary,
              character_positions_in_vocabulary=cpiv,
              batch_generator_class=LstmBatchGenerator,
              temperature=0,
              additions_to_feed_dict=[{'placeholder': 'dropout', 'value': 1.}])
