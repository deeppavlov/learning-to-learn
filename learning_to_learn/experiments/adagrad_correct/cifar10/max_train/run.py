import os
import tensorflow as tf

ROOT_HEIGHT = 5
import sys
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
from learning_to_learn.image_batch_gens import CifarBatchGenerator
from learning_to_learn.useful_functions import remove_empty_strings_from_list, convert

conf_file = sys.argv[1]
save_path = os.path.join(conf_file.split('.')[0], '%s')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
with open(conf_file, 'r') as f:
    lines = remove_empty_strings_from_list(f.read().split('\n'))
opt = lines[0]
num_runs = int(lines[1])
hps = dict()
for line in lines[2:]:
    spl = line.split()
    hps[spl[0]] = float(convert(spl[1], 'float'))

env = Environment(Mlp, CifarBatchGenerator)

train_add_feed = [
    {'placeholder': 'dropout', 'value': .9}
]
if 'momentum' in hps:
    train_add_feed.append(
        {'placeholder': 'momentum', 'value': hps['momentum']}
    )
valid_add_feed = [
    {'placeholder': 'dropout', 'value': 1.}
]

add_metrics = ['bpc', 'perplexity', 'accuracy']
VALID_SIZE = 1000

BATCH_SIZE = 32
env.build_pupil(
    batch_size=BATCH_SIZE,
    num_layers=2,
    num_hidden_nodes=[1000],
    input_shape=[3072],
    num_classes=10,
    init_parameter=hps['init_parameter'],
    additional_metrics=add_metrics,
    optimizer=opt
)

print('building is finished')
stop_specs = dict(
    type='while_progress_no_changing_parameter',
    max_no_progress_points=20,
    path_to_target_metric_storage=('valid', 'loss')
)

for run_num in range(num_runs):
    path = save_path % run_num
    learning_rate = dict(
        type='fixed',
        value=hps['learning_rate'],
    )
    env.train(
        # gpu_memory=.3,
        allow_growth=True,
        save_path=path,
        # restore_path='lstm_sample_test/scipop3_1000_bs256_11.12/checkpoints/2000',
        learning_rate=learning_rate,
        batch_size=BATCH_SIZE,
        checkpoint_steps=None,
        result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
        printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy'],
        stop=stop_specs,
        # stop=1000,
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
        additions_to_feed_dict=train_add_feed,
        validation_additions_to_feed_dict=valid_add_feed,
        no_validation=False,
        summary=False,
        add_graph_to_summary=False,
    )

    env.test(
        restore_path=os.path.join(path, 'checkpoints/best'),
        save_path=os.path.join(path, 'test'),
        additions_to_feed_dict=valid_add_feed,
        validation_datasets=dict(
            test='test'
        ),
        valid_batch_kwargs=dict(
            valid_size=VALID_SIZE
        ),
        printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy']
    )

# log_device_placement=True)
