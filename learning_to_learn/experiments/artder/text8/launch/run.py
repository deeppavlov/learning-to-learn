ROOT_HEIGHT = 5
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
from learning_to_learn.pupils.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from learning_to_learn.useful_functions import create_vocabulary

from learning_to_learn.optimizers.artder import ArtDer

import os

conf_file = sys.argv[1]
save_path = os.path.join(conf_file.split('.')[0], 'results')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

with open(conf_file, 'r') as f:
    lines = f.read().split('\n')
restore_path = lines[0]
pretrain_step = int(lines[1])

dataset_path = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'text8.txt']))
with open(dataset_path, 'r') as f:
    text = f.read()


valid_size = 500

valid_text = text[:valid_size]
train_text = text[valid_size:]

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)
print(vocabulary_size)

env = Environment(
    pupil_class=Lstm,
    meta_optimizer_class=ArtDer,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary)

add_metrics = ['bpc', 'perplexity', 'accuracy']
tmpl = os.path.join(*['..']*ROOT_HEIGHT + [restore_path, 'checkpoints', '%s'])


BATCH_SIZE = 32
NUM_UNROLLINGS = 10
env.build_pupil(
    batch_size=BATCH_SIZE,
    num_layers=1,
    num_nodes=[100],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=150,
    num_unrollings=NUM_UNROLLINGS,
    init_parameter=3.,
    num_gpus=1,
    regime='training_with_meta_optimizer',
    going_to_limit_memory=True,
    additional_metrics=add_metrics,
)

env.build_optimizer(
    regime='inference',
    additional_metrics=add_metrics,
    selection_application='shuffle',
    # selection_size=2,  # ignored if selection_application is shuffle
    # num_sel=10,
)


add_feed = [
    {'placeholder': 'dropout', 'value': .9},
    dict(
        placeholder='learning_rate',
        value=4.
    ),
    dict(
        placeholder='sel_contribution',
        value=1.
    ),
    dict(
        placeholder='selection_size',
        value=2
    ),
    dict(
        placeholder='num_sel',
        value=64
    )
]
valid_add_feed = [
    {'placeholder': 'dropout', 'value': 1.},
]

env.train(
    # gpu_memory=.3,
    num_unrollings=NUM_UNROLLINGS,
    vocabulary=vocabulary,
    with_meta_optimizer=True,
    allow_growth=True,
    save_path='debug_empty_optimizer',
    # restore_path='lstm_sample_test/scipop3_1000_bs256_11.12/checkpoints/2000',
    batch_size=BATCH_SIZE,
    checkpoint_steps=None,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=1000,
    # stop=4000,
    train_dataset_text=train_text,
    # train_dataset_text='abc',
    validation_dataset_texts=[valid_text],
    results_collect_interval=100,
    additions_to_feed_dict=add_feed,
    validation_additions_to_feed_dict=valid_add_feed,
    no_validation=False,
)
