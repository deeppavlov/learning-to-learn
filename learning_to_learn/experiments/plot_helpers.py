import sys
import os
from matplotlib import pyplot as plt, rc
from matplotlib.legend_handler import HandlerLine2D
from pathlib import Path  # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.useful_functions import synchronous_sort, create_path, remove_empty_strings_from_list, \
    convert, apply_func_to_nested, all_combs

COLORS = ['r', 'g', 'b', 'k', 'c', 'magenta', 'brown', 'darkviolet', 'pink', 'yellow', 'gray', 'orange', 'olive']
DPI = 900
FORMATS = ['pdf', 'png']
AVERAGING_NUMBER = 3

FONT = {'family': 'Verdana',
        'weight': 'normal'}


def plot_outer_legend(plot_data, description, xlabel, ylabel, xscale, yscale, file_name_without_ext, no_line):
    # print("(plot_helpers.plot_outer_legend)xlabel:", xlabel)
    rc('font', **FONT)
    plt.clf()
    plt.subplot(111)
    for_plotlib = [list(), list()]
    for label, line_data in plot_data.items():
        for_plotlib[0].append(label)
        for_plotlib[1].append(line_data)
    for_plotlib = synchronous_sort(for_plotlib, 0)
    lines = list()
    labels = list()
    if no_line:
        linestyle = 'None'
    else:
        linestyle = 'solid'
    # print("(plot_helpers.plot_outer_legend)linestyle:", linestyle)
    for idx, (label, line_data) in enumerate(zip(*for_plotlib)):
        labels.append(label)
        lines.append(
            plt.plot(
                line_data[0],
                line_data[1],
                'o-',
                color=COLORS[idx],
                label=label,
                ls=linestyle,
            )[0]
        )
    # print("(plot_helpers.plot_outer_legend)labels:", labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(xscale)
    plt.yscale(yscale)

    lgd = plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.,
        handler_map=dict(list(zip(lines, [HandlerLine2D(numpoints=1) for _ in range(len(lines))])))
    )

    for format in FORMATS:
        if format == 'pdf':
            fig_path = os.path.join(file_name_without_ext + '.pdf')
        elif format == 'png':
            fig_path = os.path.join(file_name_without_ext + '.png')
        else:
            fig_path = None
        create_path(fig_path, file_name_is_in_path=True)
        r = plt.savefig(fig_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        # print("%s %s %s %s:" % (pupil_name, res_type, regime, format), r)
    description_file = os.path.join(file_name_without_ext + '.txt')
    with open(description_file, 'w') as f:
        f.write(description)


def get_parameter_names(conf_file):
    old_dir = os.getcwd()
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    with open(conf_file, 'r') as f:
        t = f.read()
    os.chdir(old_dir)
    lines = t.split('\n')
    idx = 0
    num_lines = len(lines)
    plot_parameter_names = dict()
    while idx < num_lines and len(lines[idx]) > 0:
        spl = lines[idx].split()
        inner_name, plot_name = spl[0], spl[1:]
        plot_name = ' '.join(plot_name)
        plot_parameter_names[inner_name] = plot_name
        idx += 1
    # print("(plot_helpers.get_parameter_names)plot_parameter_names:", plot_parameter_names)
    return plot_parameter_names


def parse_eval_dir(eval_dir):
    dir_sets = eval_dir.split(':')
    dir_sets_prepared = list()
    for dir_set in dir_sets:
        dir_sets_prepared.append(dir_set.split(','))
    eval_dirs = [os.path.join(*comb) for comb in all_combs(dir_sets_prepared)]
    # print("(plot_helpers.parse_eval_dir)eval_dirs:", eval_dirs)
    filtered = list()
    for dir_ in eval_dirs:
        if os.path.exists(dir_):
            filtered.append(dir_)
    # print("(plot_helpers.parse_eval_dir)filtered:", filtered)
    return filtered


def plot_hp_search(
        eval_dir,
        plot_dir,
        hp_plot_order,
        hp_names_file,
        metric_scales,
        xscale,
        no_line,
):
    plot_parameter_names = get_parameter_names(hp_names_file)
    changing_hp = hp_plot_order[-1]
    with open(os.path.join(eval_dir, 'hp_layout.txt'), 'r') as f:
        hp_save_order = f.read().split()

    create_path(plot_dir)

    eval_dir_contents = os.listdir(eval_dir)
    eval_dir_contents.remove('hp_layout.txt')
    for entry in eval_dir_contents:
        if 'launch_log' in entry:
            eval_dir_contents.remove(entry)
        if 'plot' in entry:
            eval_dir_contents.remove(entry)

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
        regime = regime.split('.')[0]
        if regime not in regimes:
            regimes.append(regime)

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

    # print("(plot_helpers.plot_hp_search)regimes:", regimes)
    for hp_value_file, result_dir in zip(hp_value_files, result_dirs):
        hp_value_file = os.path.join(eval_dir, hp_value_file)
        with open(hp_value_file, 'r') as f:
            hp_v = [convert(v, type_) for v, type_ in zip(f.read().split('\n')[0].split(), hp_types)]
        hp_values = [hp_v[hp_map[hp]] for hp in hp_plot_order]
        print("(plot_hp_search)hp_values:", hp_values)
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
                    # print(pupil_name, res_type, regime, fixed_hps_tuple, line_hp_value, changing_hp_value)
                    r[0].append(changing_hp_value)
                    r[1].append(mean)

    # print("(plot_helpers.plot_hp_search)for_plotting:", for_plotting)
    for_plotting = apply_func_to_nested(for_plotting, lambda x: synchronous_sort(x, 0), (dict,))

    file_with_hp_layout_description = os.path.join(plot_dir, 'plot_hp_layout.txt')
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

    # print("(plot_hp_search)plot_parameter_names:", plot_parameter_names)
    xlabel = plot_parameter_names[changing_hp]

    for pupil_name in pupil_names:
        for res_type in result_types:
            if res_type in plot_parameter_names:
                ylabel = plot_parameter_names[res_type]
            else:
                ylabel = res_type
            if res_type in metric_scales:
                yscale = metric_scales[res_type]
            else:
                yscale = 'linear'
            for regime in regimes:
                path = os.path.join(plot_dir, pupil_name, res_type, regime)
                create_path(path)
                counter = 0
                d = for_plotting[pupil_name][res_type][regime]
                on_descriptions = dict()
                for fixed_hps_tuple, plot_data in d.items():
                    plot_data_on_labels = dict()
                    for line_hp_value, line_data in plot_data.items():
                        plot_data_on_labels['{:.0e}'.format(line_hp_value)] = line_data
                    on_descriptions[tmpl[:-1] % fixed_hps_tuple] = plot_data_on_labels
                    # print("(plot_helpers.plot_hp_search)plot_data:", plot_data)
                    file_name_without_ext = os.path.join(path, str(counter))
                    for description, plot_data in on_descriptions.items():
                        plot_outer_legend(
                            plot_data, description, xlabel, ylabel, xscale, yscale, file_name_without_ext, no_line
                        )
                        counter += 1
