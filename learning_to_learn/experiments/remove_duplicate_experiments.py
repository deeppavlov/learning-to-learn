import sys
import os
from shutil import copyfile, copytree

from pathlib import Path  # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.useful_functions import is_int, create_path

source_dir = sys.argv[1]
dest_dir = sys.argv[2]
create_path(dest_dir)

contents = os.listdir(source_dir)

exp_description_files = list()
for entry in contents:
    if entry[-4:] == '.txt' and is_int(entry[:-4]):
        exp_description_files.append(entry)

filtered = list()
hp_sets = list()
for file_name in exp_description_files:
    with open(os.path.join(source_dir, file_name), 'r') as f:
        hp_set = tuple(f.read().split('\n')[0].split())
        if hp_set not in hp_sets:
            hp_sets.append(hp_set)
            filtered.append(file_name)

for file_name in filtered:
    copyfile(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
    results_file = file_name[:-4]
    copytree(os.path.join(source_dir, results_file), os.path.join(dest_dir, results_file))

copyfile(os.path.join(source_dir, 'hp_layout.txt'), os.path.join(dest_dir, 'hp_layout.txt'))
copyfile(os.path.join(source_dir, 'launch_log.txt'), os.path.join(dest_dir, 'launch_log.txt'))
if 'plot.conf' in contents:
    copyfile(os.path.join(source_dir, 'plot.conf'), os.path.join(dest_dir, 'plot.conf'))
