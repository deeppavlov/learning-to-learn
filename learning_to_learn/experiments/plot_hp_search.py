import sys
import os
import matplotlib.pyplot as plt

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

AVERAGING_NUMBER = 3
COLORS = ['r', 'g', 'b', 'k', 'c', 'magenta', 'brown', 'darkviolet', 'pink', 'yellow', 'gray', 'orange', 'olive']
LABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
DPI = 900
FORMATS = ['pdf', 'png']

eval_dir = sys.argv[1]
path_for_plot_saving = sys.argv[2]
hp_plot_order = sys.argv[3:]
changing_hp = hp_plot_order[-1]

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

with open(os.path.join(eval_dir, 'hp_layout.txt'), 'r') as f:
    hp_save_order = f.read().split()

create_path(path_for_plot_saving)

eval_dir_contents = os.listdir(eval_dir)
eval_dir_contents.remove('hp_layout.txt')
for entry in eval_dir_contents:
    if 'launch_log' in entry:
        eval_dir_contents.remove(entry)

plot_conf_file = 'plot.conf'
plot_parameter_names = dict()
scales = dict()
if plot_conf_file in eval_dir_contents:
    path_to_plot_parameter_names_file = os.path.join(eval_dir, plot_conf_file)
    with open(path_to_plot_parameter_names_file, 'r') as f:
        t = f.read()
    lines = t.split('\n')
    num_lines = len(lines)
    idx = 0
    while idx < num_lines and len(lines[idx]) > 0:
        spl = lines[idx].split()
        inner_name, plot_name = spl[0], spl[1:]
        plot_name = ' '.join(plot_name)
        plot_parameter_names[inner_name] = plot_name
        idx += 1
    idx += 1
    while idx < num_lines and len(lines[idx]) > 0:
        [hp, scale] = lines[idx].split()
        scales[hp] = scale
        idx += 1

    eval_dir_contents.remove(plot_conf_file)

for hp_name in hp_plot_order:
    if hp_name not in plot_parameter_names:
        plot_parameter_names[hp_name] = hp_name
    if hp_name not in scales:
        scales[hp_name] = 'linear'

hp_value_files = list()
result_dirs = list()
for entry in eval_dir_contents:
    if entry[-4:] == '.txt':
        hp_value_files.append(entry)
    else:
        result_dirs.append(entry)

hp_value_files = sorted(hp_value_files, key=lambda elem: int(elem[:-4]))
result_dirs = sorted(result_dirs, key=lambda elem: int(elem))

pupil_names = os.listdir(os.path.join(eval_dir, result_dirs[0]))

res_files = os.listdir(os.path.join(eval_dir, result_dirs[0], pupil_names[0]))

result_types = list()
regimes = list()
for res_file in res_files:
    [res_type, regime] = res_file.split('_')
    if res_type not in result_types:
        result_types.append(res_type)
    if regime not in regimes:
        regimes.append(regime.split('.')[0])

hp_map = list()
# print("(plot_hp_search)hp_save_order:", hp_save_order)
for hp in hp_plot_order:
    hp_map.append((hp, hp_save_order.index(hp)))
hp_map = dict(hp_map)

for_plotting = dict()
for pupil_name in pupil_names:
    for_plotting[pupil_name] = dict()
    for res_type in result_types:
        for_plotting[pupil_name][res_type] = dict()
        for regime in regimes:
            for_plotting[pupil_name][res_type][regime] = dict()

file_for_hp_types = os.path.join(eval_dir, hp_value_files[0])
# print("(plot_hp_search)file_for_hp_types:", file_for_hp_types)
with open(file_for_hp_types, 'r') as f:
    hp_types = f.read().split('\n')[1].split()

for hp_value_file, result_dir in zip(hp_value_files, result_dirs):
    hp_value_file = os.path.join(eval_dir, hp_value_file)
    with open(hp_value_file, 'r') as f:
        hp_v = [convert(v, type_) for v, type_ in zip(f.read().split('\n')[0].split(), hp_types)]
    hp_values = [hp_v[hp_map[hp]] for hp in hp_plot_order]
    # print("(plot_hp_search)hp_values:", hp_values)
    fixed_hps_tuple = tuple(hp_values[:-2])
    if len(hp_values) > 1:
        line_hp_value = hp_values[-2]
    else:
        line_hp_value = None
    changing_hp_value = hp_values[-1]
    for pupil_name in pupil_names:
        for res_type in result_types:
            for regime in regimes:
                d = for_plotting[pupil_name][res_type][regime]
                if fixed_hps_tuple not in d:
                    d[fixed_hps_tuple] = dict()
                d = d[fixed_hps_tuple]
                if line_hp_value not in d:
                    d[line_hp_value] = [list(), list()]
                r = d[line_hp_value]
                file_name = res_type + '_' + regime + '.txt'
                file_with_data = os.path.join(eval_dir, result_dir, pupil_name, file_name)
                with open(file_with_data, 'r') as f:
                    t = f.read()
                lines = t.split('\n')
                lines = remove_empty_strings_from_list(lines)
                s = 0
                # print("(plot_hp_search)lines:", lines)
                for i in range(AVERAGING_NUMBER):
                    s += float(lines[-i].split()[-1])
                mean = s / AVERAGING_NUMBER
                r[0].append(changing_hp_value)
                r[1].append(mean)

for_plotting = apply_func_to_nested(for_plotting, lambda x: synchronous_sort(x, 0), (dict,))

file_with_hp_layout_description = os.path.join(path_for_plot_saving, 'plot_hp_layout.txt')
num_of_hps = len(hp_plot_order)
if num_of_hps > 2:
    tmpl = '%s ' * (num_of_hps - 3) + '%s\n'
else:
    tmpl = '\n'
if num_of_hps > 1:
    line_hp_name = hp_plot_order[-2] + '\n'
else:
    line_hp_name = '\n'
with open(file_with_hp_layout_description, 'w') as f:
    f.write('fixed hyper parameters: ' + tmpl % tuple(hp_plot_order[:-2]))

    f.write('line hyper parameter: ' + line_hp_name)
    f.write('changing hyper parameter: ' + changing_hp)

# print("(plot_hp_search)scales:", scales)
for pupil_name in pupil_names:
    for res_type in result_types:
        for regime in regimes:
            path = os.path.join(path_for_plot_saving, pupil_name, res_type, regime)
            create_path(path)
            counter = 0
            d = for_plotting[pupil_name][res_type][regime]
            for fixed_hps_tuple, plot_data in d.items():
                plt.clf()
                line_hp_values = list()
                for_plotlib = [list(), list()]
                for line_hp_value, line_data in plot_data.items():
                    for_plotlib[0].append(line_hp_value)
                    for_plotlib[1].append(line_data)
                for_plotlib = synchronous_sort(for_plotlib, 0)
                for idx, (line_hp_value, line_data) in enumerate(zip(*for_plotlib)):
                    line_hp_values.append(line_hp_value)
                    plt.plot(line_data[0], line_data[1], 'o-', color=COLORS[idx], label='{:.0e}'.format(line_hp_value))
                plt.xlabel(plot_parameter_names[changing_hp])
                plt.ylabel(res_type)
                if scales[changing_hp] == 'log':
                    plt.xscale('log')
                plt.legend(loc='center left')
                for format in FORMATS:
                    if format == 'pdf':
                        fig_path = os.path.join(path, str(counter) + '.pdf')
                        r = plt.savefig(fig_path)
                    if format == 'png':
                        fig_path = os.path.join(path, str(counter) + '.png')
                        r = plt.savefig(fig_path, dpi=DPI)
                    # print("%s %s %s %s:" % (pupil_name, res_type, regime, format), r)
                fixed_hps_file = os.path.join(path, str(counter) + '.txt')
                with open(fixed_hps_file, 'w') as f:
                    f.write(tmpl[:-1] % fixed_hps_tuple)
                counter += 1
