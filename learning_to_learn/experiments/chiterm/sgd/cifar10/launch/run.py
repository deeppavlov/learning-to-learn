ROOT_HEIGHT = 6
import sys
import tensorflow as tf
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

from learning_to_learn.optimizers.chiterm import ChiTerm

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

conf_file = sys.argv[1]
save_path = os.path.join(conf_file.split('.')[0], 'results')

with open(conf_file, 'r') as f:
    lines = f.read().split('\n')
restore_path = lines[0]

data_dir = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'mnist']))

env = Environment(
    pupil_class=Mlp,
    meta_optimizer_class=ChiTerm,
    batch_generator_classes=CifarBatchGenerator,
)
VALID_SIZE = 1000
add_metrics = ['bpc', 'perplexity', 'accuracy']


BATCH_SIZE = 32
env.build_pupil(
    batch_size=BATCH_SIZE,
    num_layers=1,
    num_hidden_nodes=[],
    input_shape=[3072],
    num_classes=10,
    init_parameter=1.,
    additional_metrics=add_metrics,
    regime='training_with_meta_optimizer',
)

env.build_optimizer(
    regime='inference',
    additional_metrics=add_metrics,
    chi_application='exp',
)


print('building is finished')
add_feed = [
    {'placeholder': 'dropout', 'value': .9},
    dict(
        placeholder='learning_rate',
        value=4.
    ),
    dict(
        placeholder='chi_contribution',
        value=.01
    )
]
valid_add_feed = [
    {'placeholder': 'dropout', 'value': 1.},
]

tf.set_random_seed(1)
env.train(
    # gpu_memory=.3,
    allow_growth=True,
    save_path='debug_early_stop',
    with_meta_optimizer=True,
    # restore_path='lstm_sample_test/scipop3_1000_bs256_11.12/checkpoints/2000',
    batch_size=32,
    checkpoint_steps=None,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    stop=1000,
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
    results_collect_interval=100,
    additions_to_feed_dict=add_feed,
    validation_additions_to_feed_dict=valid_add_feed,
    no_validation=False
)