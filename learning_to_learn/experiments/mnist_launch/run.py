import tensorflow as tf

ROOT_HEIGHT = 3
import sys
import os
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[ROOT_HEIGHT]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
from learning_to_learn.environment import Environment
from learning_to_learn.pupils.mlp_for_meta import MlpForMeta as Mlp
from learning_to_learn.image_batch_gens import MnistBatchGenerator


conf_file = sys.argv[1]
save_path = '.'.join(conf_file.split('.')[:-1])
with open(conf_file, 'r') as f:
    lines = f.read().split('\n')
optimizer = lines[0]
freq = int(lines[1])
stop = int(lines[2])
ip = float(lines[3])
lr = float(lines[4])

data_dir = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'mnist']))

env = Environment(Mlp, MnistBatchGenerator)

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

BATCH_SIZE = 32
env.build_pupil(
    batch_size=BATCH_SIZE,
    num_layers=1,
    num_hidden_nodes=[],
    input_shape=[784],
    num_classes=10,
    init_parameter=ip,
    additional_metrics=add_metrics,
    optimizer=optimizer,
)

print('building is finished')
stop_specs = dict(
    type='while_progress',
    max_no_progress_points=10,
    changing_parameter_name='learning_rate',
    path_to_target_metric_storage=('valid', 'loss')
)
learning_rate = dict(
    type='adaptive_change',
    max_no_progress_points=10,
    decay=.5,
    init=lr,
    path_to_target_metric_storage=('valid', 'loss')
)
env.train(
    # gpu_memory=.3,
    allow_growth=True,
    save_path='debug_early_stop',
    # restore_path='lstm_sample_test/scipop3_1000_bs256_11.12/checkpoints/2000',
    learning_rate=learning_rate,
    batch_size=BATCH_SIZE,
    checkpoint_steps=None,
    result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
    # stop=stop_specs,
    stop=stop,
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
    results_collect_interval=freq,
    additions_to_feed_dict=add_feed,
    validation_additions_to_feed_dict=valid_add_feed,
    no_validation=False,
)
# log_device_placement=True)