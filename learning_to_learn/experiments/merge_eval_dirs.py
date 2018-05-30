import sys
import os
import shutil
from pathlib import Path  # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.useful_functions import get_num_exps_and_res_files

dest_dir = sys.argv[1]

source_dirs = sys.argv[2:]

_, biggest_idx, _ = get_num_exps_and_res_files(dest_dir)

for source_dir in source_dirs:
    _, _, pairs = get_num_exps_and_res_files(source_dir)
    for pair in pairs:
        file_name = os.path.split(pair[0])[-1]
        new_prefix = str(int(file_name[:-4]) + 1 + biggest_idx)
        file_dest = os.path.join(dest_dir, new_prefix + '.txt')
        dir_dest = os.path.join(dest_dir, new_prefix)
        shutil.copyfile(pair[0], file_dest)
        shutil.copytree(pair[1], dir_dest)
    _, biggest_idx, _ = get_num_exps_and_res_files(dest_dir)
