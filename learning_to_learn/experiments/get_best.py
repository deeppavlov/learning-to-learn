import sys
import os

from pathlib import Path  # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.useful_functions import get_optimizer_evaluation_results, get_pupil_evaluation_results, \
    get_metric_names_and_regimes_from_optimizer_eval_dir, get_pupil_names_from_eval_dir, \
    get_dataset_names_from_eval_dir, get_hp_names_from_optimizer_eval_dir, \
    get_hp_and_metric_names_from_pupil_eval_dir, get_best, print_hps
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "eval_dir",
    help="Path to evaluation directory containing experiment results"
)


parser.add_argument(
    '-m',
    "--model",
    help="Optimized model type. It can be 'pupil' or optimizer or 'optimizer'. Default is 'optimizer'",
    default='optimizer',
)
parser.add_argument(
    '-tm',
    "--target_metrics",
    help="Name of metrics to be maximized. Available 'bpc, 'perplexity', 'loss', 'accuracy. Default is all",
    default=None,
)
parser.add_argument(
    '-pn',
    '--pupil_names',
    help="Examined pupil names. Default is all present.",
    default=None
)
parser.add_argument(
    '-rg',
    '--regimes',
    help="Examined regimes. Default is all present.",
    default=None
)
parser.add_argument(
    '-dn',
    '--dataset_names',
    help="Examined datasets. Default is all available.",
    default=None
)
args = parser.parse_args()


AVERAGING_NUMBER = 3
eval_dir = args.eval_dir
model = args.model
if model == 'pupil':
    hp_names, metrics = get_hp_and_metric_names_from_pupil_eval_dir(eval_dir)
    dataset_names = get_dataset_names_from_eval_dir(eval_dir)
    if args.dataset_names is not None:
        dataset_names = args.dataset_names.split(',')
    for_plotting = get_pupil_evaluation_results(args.eval_dir, hp_names)

elif model == 'optimizer':
    hp_names = get_hp_names_from_optimizer_eval_dir(eval_dir)
    metrics, regimes = get_metric_names_and_regimes_from_optimizer_eval_dir(eval_dir)
    if args.regimes is not None:
        regimes = args.regimes.split(',')
    pupil_names = get_pupil_names_from_eval_dir(eval_dir)
    if args.pupil_names is not None:
        pupil_names = args.pupil_names.split(',')
    for_plotting = get_optimizer_evaluation_results(eval_dir, hp_names, AVERAGING_NUMBER)

if args.target_metrics is not None:
    metrics = args.target_metrics.split(',')

best = get_best(for_plotting, model)
indents = [4, 8, 12]
print(best)
if model == 'pupil':
    for dataset_name in dataset_names:
        print('dataset:', dataset_name)
        for metric in metrics:
            b = best[dataset_name][metric]
            print(' ' * indents[0] + metric + ':', b[1])
            print_hps(hp_names, b[0], indents[1])
else:
    for pupil_name in pupil_names:
        print('pupil name:', pupil_name)
        for metric in metrics:
            print(' ' * indents[0] + metric + ':')
            for regime in regimes:
                b = best[pupil_name][metric][regime]
                print(' ' * indents[1] + regime + ' result:', regime)
                print(' ' * indents[2] + 'result:', b[1])
                print_hps(hp_names, b[0], indents[2])
