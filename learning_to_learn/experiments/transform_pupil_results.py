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

from learning_to_learn.experiments.plot_helpers import get_parameter_names, plot_hp_search_optimizer, \
    plot_hp_search_pupil, parse_metric_scales_str
from learning_to_learn.useful_functions import perform_transformation, parse_path_comb
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "eval_file",
    help="Path to evaluation file containing experiment results. \nTo process several evaluation files"
         " use following format '<path1>,<path2>,...<pathi>:<pathi+1>,..:..'.\nAll possible combinations of sets"
         " separated by colons (in specified order) will be processed. \nYou have to provide paths relative to "
         "script. Edge characters of <path> can't be '/'"
)

parser.add_argument(
    "-t",
    "--transformation",
    help="transformation to perform. Available options '1_minus_x'"
)

parser.add_argument(
    '-ed',
    "--new_eval_dir",
    help='new_eval_dir name. Default is eval_fix'
)

parser.add_argument(
    '-on',
    '--old_name',
    help="old parameter names",
)
parser.add_argument(
    '-nn',
    '--new_name',
    help='new parameter names',
)
parser.add_argument(
    '-tp',
    '--types',
    help="Types of parameters. default is float for all",
    default=None,
)
args = parser.parse_args()

transformations = args.transformation.split(',')

old_names = args.old_name.split(',')

new_names = args.new_name.split(',')
num_params = len(old_names)

res_files = parse_path_comb(args.eval_file)

for f in res_files:
    perform_transformation(f, args.new_eval_dir, old_names, new_names, args.types, transformations)
