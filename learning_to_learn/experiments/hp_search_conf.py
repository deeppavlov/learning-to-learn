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

from learning_to_learn.useful_functions import create_path, parse_path_comb, get_points_from_range, get_tmpl
parser = argparse.ArgumentParser()

parser.add_argument(
    "confs",
    help="Paths to created configs used for hp search. \nTo process several configs"
         " use following format '<path1>,<path2>,...<pathi>:<pathi+1>,..:..'.\nAll possible combinations of sets"
         " separated by colons (in specified order) will be processed. Combinations are formed in the following way: "
         "from each set one name is chosen\nYou have to provide paths relative to "
         "script. Edge characters of <path> can't be '/'"
)
parser.add_argument(
    '-hp',
    "--hyper_parameters",
    help="Hyper parameters in conf. You have to specify hyper parameter name and type separating them with comma. "
         "To pass several hyper parameters to script separate thei specs with colons"
)
parser.add_argument(
    '-s',
    "--span",
    help="Range of hyper parameters values. Specify start, end, number of points and scale, separating them with commas"
         ". Ranges for  several hps separated by colons. Example\n1e-5,1,20,log"
)
parser.add_argument(
    '-n',
    '--num_repeats',
    help="Number of times this conf will be repeated",
    default=1,
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Regulate verbosity",
    action='store_true'
)
parser.add_argument(
    '-ng',
    '--no_git',
    help="don't add confs to git",
    action='store_true',
)

args = parser.parse_args()

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

confs = parse_path_comb(args.confs, filter_=False)

hp_names_with_types = args.hyper_parameters.split(':')
hp_names = list()
types = list()
for s in hp_names_with_types:
    spl = s.split(',')
    hp_names.append(spl[0])
    types.append(spl[1])

span = args.span.split(':')

points = list()
for string in span:
    points.append(
        get_points_from_range(string)
    )

indent = 4
for conf in confs:
    create_path(conf, file_name_is_in_path=True)
    tmpl = get_tmpl(hp_names)
    file_string = ''

    file_string += tmpl % tuple(hp_names) + '\n'
    file_string += tmpl % tuple(types) + '\n'
    for values in points:
        file_string += get_tmpl(values) % tuple(values) + '\n'
    file_string += '%s' % args.num_repeats
    with open(conf, 'w') as f:
        f.write(file_string)
    if args.verbose:
        print('\n' + ' '*indent + conf)
        print(file_string)

if not args.no_git:
    for conf in confs:
        command = 'git add %s' % conf
        os.system(command)
