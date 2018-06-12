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

from learning_to_learn.useful_functions import convert, apply_func_to_nested, synchronous_sort, create_path, \
    remove_empty_strings_from_list

from learning_to_learn.experiments.plot_helpers import plot_outer_legend, get_parameter_names
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
    "-pd",
    "--plot_dir",
    help="Path to directory where plots are going to be saved",
    default=None,
)
parser.add_argument(
    "-xs",
    "--xscale",
    help="Axes scale. It can be log or linear. Default is linear",
    default='linear',
)
parser.add_argument(
    "-ms",
    "--metric_scales",
    help="Scales for metrics. Available metrics are 'accuracy', 'bpc', 'loss', 'perplexity'. "
         "Scales are provided in following format <metric>:<scale>,<metric>:<scale>. "
         "Default is linear"
)
parser.add_argument(
    "-nlfp",
    "--num_lines_for_plot",
    help="Number of lines per one plot. Default is all lines."
)
parser.add_argument(
    "-nl",
    "--no_line",
    help="Do not link dots with line. Default is True",
    action='store_true'
)
parser.add_argument(
    '-hpnf',
    "--hp_names_file",
    help="File with hyper parameter names. All available files are in the same directory with this script",
    default='hp_plot_names_english.conf'
)
args = parser.parse_args()

AVERAGING_NUMBER = 3

eval_dir = args.eval_dir
if args.plot_dir is None:
    plot_dir = os.path.join(os.path.split(eval_dir)[:-1], 'plots')
else:
    plot_dir = args.plot_dir