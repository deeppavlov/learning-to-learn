ROOT_HEIGHT = 3
import sys
from pathlib import Path
import tensorflow as tf
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[ROOT_HEIGHT]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.environment import Environment
from learning_to_learn.pupils.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator
from learning_to_learn.useful_functions import create_vocabulary, remove_empty_strings_from_list, convert

import os

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

dataset_path = os.path.join(*(['..']*ROOT_HEIGHT + ['datasets', 'text8.txt']))
with open(dataset_path, 'r') as f:
    text = f.read()


valid_size = 2000
test_size = 100000
test_text = text[:test_size]
valid_text = text[test_size:valid_size+test_size]
train_text = text[valid_size+test_size:]

vocabulary = create_vocabulary(text)
vocabulary_size = len(vocabulary)
print(vocabulary_size)

env = Environment(
    pupil_class=Lstm,
    batch_generator_classes=BatchGenerator,
    vocabulary=vocabulary)

add_metrics = ['bpc', 'perplexity', 'accuracy']
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

dataset_name = 'valid'

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
    num_unrollings=10,
    num_gpus=1,
    init_parameter=hps['init_parameter'],
    regime='autonomous_training',
    additional_metrics=add_metrics,
    going_to_limit_memory=True,
    optimizer=opt,
)

stop_specs = dict(
    type='while_progress',
    max_no_progress_points=20,
    changing_parameter_name='learning_rate',
    path_to_target_metric_storage=('valid', 'loss')
)

for run_num in range(num_runs):
    path = save_path % run_num
    env.train(
        allow_growth=True,
        save_path=path,
        result_types=['loss', 'bpc', 'perplexity', 'accuracy'],
        additions_to_feed_dict=train_add_feed,
        # pupil_restore_paths=['debug_empty_meta_optimizer/not_learning_issue_es20_nn20/checkpoints/0'],
        # stop=1000,
        stop=stop_specs,
        vocabulary=vocabulary,
        num_unrollings=NUM_UNROLLINGS,
        results_collect_interval=100,
        learning_rate=dict(
            type='adaptive_change',
            decay=.5,
            init=hps['learning_rate'],
            max_no_progress_points=20,
            path_to_target_metric_storage=('valid', 'loss'),
        ),
        # opt_inf_results_collect_interval=1,
        summary=False,
        add_graph_to_summary=False,
        train_dataset_text=train_text,
        validation_datasets=dict(valid=valid_text),
        batch_size=BATCH_SIZE
    )

    env.test(
        restore_path=os.path.join(path, 'checkpoints/best'),
        save_path=os.path.join(path, 'test'),
        vocabulary=vocabulary,
        additions_to_feed_dict=valid_add_feed,
        validation_dataset_texts=[test_text],
        valid_batch_kwargs=dict(
            vocabulary=vocabulary
        ),
        printed_result_types=['perplexity', 'loss', 'bpc', 'accuracy']
    )
