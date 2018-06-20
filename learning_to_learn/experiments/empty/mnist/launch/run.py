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
from learning_to_learn.pupils.mlp_for_meta import MlpForMeta as Mlp
from learning_to_learn.optimizers.empty import Empty
from learning_to_learn.image_batch_gens import MnistBatchGenerator

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
data_dir = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'mnist']))

env = Environment(
    pupil_class=Mlp,
    meta_optimizer_class=Empty,
    batch_generator_classes=MnistBatchGenerator,
)

add_metrics = ['bpc', 'perplexity', 'accuracy']
tmpl = os.path.join(*['..']*ROOT_HEIGHT + [restore_path, 'checkpoints', '%s'])


BATCH_SIZE = 32
env.build_pupil(
    batch_size=BATCH_SIZE,
    num_layers=2,
    num_hidden_nodes=[1000],
    input_shape=[784],
    num_classes=10,
    init_parameter=3.,
    additional_metrics=add_metrics,
    regime='training_with_meta_optimizer'
)

env.build_optimizer(
    regime='inference',
    additional_metrics=add_metrics,
    get_omega_and_beta=True,
    matrix_mod='omega',
)


add_feed = [
    {'placeholder': 'dropout', 'value': .9},
    dict(
        placeholder='learning_rate',
        value=.01
    )
]
valid_add_feed = [
    {'placeholder': 'dropout', 'value': 1.},
]

env.train(
    # gpu_memory=.3,
    # gpu_memory=.3,
    allow_growth=True,
    save_path='debug_empty_optimizer',
    with_meta_optimizer=True,
    # restore_path='lstm_sample_test/scipop3_1000_bs256_11.12/checkpoints/2000',
    batch_size=32,
    checkpoint_steps=None,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    # stop=stop_specs,
    stop=4000,
    train_dataset=dict(
        train='train'
    ),
    train_batch_kwargs=dict(
        data_dir=data_dir
    ),
    valid_batch_kwargs=dict(
        data_dir=data_dir
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