import sys
from pathlib import Path  # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.useful_functions import get_missing_hp_sets

conf_file = sys.argv[1]
eval_dir = sys.argv[2]
model = sys.argv[3]

missing_hp_sets = get_missing_hp_sets(conf_file, eval_dir, model)
num_missing = len(missing_hp_sets)
for idx, hp_set in enumerate(missing_hp_sets):
    print(idx)
    for hp_name, hp_value in hp_set.items():
        print(hp_name, hp_value)
    if idx < num_missing - 1:
        print()
