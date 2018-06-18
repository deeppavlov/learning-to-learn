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

from learning_to_learn.useful_functions import create_path, parse_path_comb, get_points_from_range, get_tmpl, \
    remove_empty_strings_from_list, perform_cut, perform_sym_cut
parser = argparse.ArgumentParser()

parser.add_argument(
    "path",
    help="Path to dir with confs.\nYou have to provide paths relative to "
         "script. Edge characters of <path> can't be '/'"
)
parser.add_argument(
    '-o',
    "--old",
    help="Old config name. relative to 'path'"
)
parser.add_argument(
    '-n',
    "--new",
    help="New config name"
)
parser.add_argument(
    '-hp',
    "--hyper_parameters",
    help="Hyper parameters to replace. Use the following format <hp_name>[<span>]:..."
)

parser.add_argument(
    '-nr',
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

path = args.path
print(args.old)
new = os.path.join(path, args.new)
old = os.path.join(path, args.old)

with open(old, 'r') as f:
    lines = f.read().split('\n')

hp_names_str = lines[0]
hp_names = remove_empty_strings_from_list(hp_names_str.split())
types = lines[1]

repl_hps, points = list(), list()
for hp_str in remove_empty_strings_from_list(args.hyper_parameters.split(':')):
    spl = hp_str.split('[')
    repl_hps.append(spl[0])
    [specs, application] = spl[1].split(']')
    if application == 'list':
        points.append(specs.split(','))
    elif application == 'range':
        points.append(get_points_from_range(specs))
    elif application == 'cut':
        pidx = hp_names.index(spl[0])
        points.append(perform_cut(lines[2+pidx], specs))
    elif application == 'symcut':
        pidx = hp_names.index(spl[0])
        points.append(perform_sym_cut(lines[2+pidx], specs))


repl = dict(list(zip(repl_hps, points)))

file_string = ''
file_string += hp_names_str + '\n'
file_string += types + '\n'
for idx, name in enumerate(hp_names):
    if name in repl:
        file_string += get_tmpl(repl[name]) % tuple(repl[name]) + '\n'
    else:
        file_string += lines[2+idx] + '\n'
file_string += '%s' % args.num_repeats
indent = 4
with open(new, 'w') as f:
    f.write(file_string)
if args.verbose:
    print('\n' + ' ' * indent + new)
    print(file_string)


if not args.no_git:
    command = 'git add %s' % new
    os.system(command)
