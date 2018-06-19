import os
import argparse

import sys
from pathlib import Path  # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.useful_functions import create_path, parse_path_comb, get_tmpl, \
    remove_empty_strings_from_list, pupil_test_results_summarize, stats_string_for_all_models
parser = argparse.ArgumentParser()

parser.add_argument(
    "model_paths",
    help="Paths to dirs with model results\nYou have to provide paths relative to "
         "script. Edge characters of <path> can't be '/'"
)
parser.add_argument(
    '-o',
    "--output",
    help="Output file name. Path relative to this script."
)
parser.add_argument(
    '-i',
    "--inter_path",
    help="Path between model folder and results folder. default is 'test'",
    default='test',
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Regulate verbosity",
    action='store_true'
)
args = parser.parse_args()

model_paths = parse_path_comb(args.model_paths, filter_=False)

results = dict()

for model_path in model_paths:
    results[model_path] = pupil_test_results_summarize(model_path, args.inter_path)

string = stats_string_for_all_models(results, 2)

create_path(args.output, file_name_is_in_path=True)

with open(args.output, 'w') as f:
    f.write(string)

if args.verbose:
    print(string)
