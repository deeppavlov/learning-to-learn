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
    get_hp_and_metric_names_from_pupil_eval_dir, get_best
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "hp_order",
    help="Order of hyper parameters. Hyper parameter names as in run config separated by commas without spaces"
)
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
    "target_metrics",
    help="Name of metrics to be maximized. Available 'bpc, 'perplexity', 'loss', 'accuracy. Default is all",
    default=None,
)
parser.add_argument(
    '-rg',
    '--regimes',
    help="Examined regimes. Default is both validation and train.",
    default=None
)
parser.add_argument(
    '-rg',
    '--dataset_names',
    help="Examined datasets. Default is all available.",
    default=None
)
args = parser.parse_args()


target_metrics = args.target_metrics.split(',')
regimes = args.regimes.split(',')

AVERAGING_NUMBER = 3
eval_dir = args.eval_dir
model = args.model
if model == 'pupil':
    hp_names, metrics = get_hp_and_metric_names_from_pupil_eval_dir(eval_dir)
    if args.target_metrics is not None:
        metrics = args.target_metrics
    dataset_names = get_dataset_names_from_eval_dir(eval_dir)
    if args.dataset_names is not None:
        dataset_names = args.dataset_names
    for_plotting = get_pupil_evaluation_results(args.eval_dir, hp_names)

elif model == 'optimizer':
    hp_names = get_hp_names_from_optimizer_eval_dir(eval_dir)
    metrics, regimes = get_metric_names_and_regimes_from_optimizer_eval_dir(eval_dir)
    if args.target_metrics is not None:
        metrics = args.target_metrics
    if args.regimes is not None:
        regimes = args.regimes
    pupil_names = get_pupil_names_from_eval_dir(eval_dir)
    if args.pupil_names is not None:
        pupil_names = args.pupil_names
    for_plotting = get_optimizer_evaluation_results(eval_dir, hp_names, AVERAGING_NUMBER)

best = get_best(for_plotting, model)

