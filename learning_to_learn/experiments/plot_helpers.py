import random
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

from learning_to_learn.useful_functions import synchronous_sort, create_path, get_pupil_evaluation_results, \
    BadFormattingError, all_combs, get_optimizer_evaluation_results

COLORS = [
    'r', 'g', 'b', 'k', 'c', 'magenta', 'brown',
    'darkviolet', 'pink', 'yellow', 'gray', 'orange', 'olive',

]
DPI = 900
FORMATS = ['pdf', 'png']
AVERAGING_NUMBER = 3

FONT = {'family': 'Verdana',
        'weight': 'normal'}


def get_linthreshx(lines):
    left = None
    right = None
    for line_data in lines:
        for x in line_data[0]:
            if x < 0 and (left is None or (left is not None and x > left)):
                left = x
            if x > 0 and (right is None or (right is not None and x < right)):
                right = x
    if left is None:
        if right is None:
            thresh = 1.
        else:
            thresh = abs(right)
    else:
        if right is None:
            thresh = abs(left)
        else:
            thresh = max(abs(left), abs(right))
    return thresh


def plot_outer_legend(plot_data, description, xlabel, ylabel, xscale, yscale, file_name_without_ext, no_line):
    # print("(plot_helpers.plot_outer_legend)xlabel:", xlabel)
    rc('font', **FONT)
    plt.clf()
    plt.subplot(111)
    for_plotlib = [list(), list()]
    for label, line_data in plot_data.items():
        for_plotlib[0].append(label)
        for_plotlib[1].append(line_data)
    for_plotlib = synchronous_sort(for_plotlib, 0, lambda_func=lambda x: float(x))
    lines = list()
    labels = list()
    if no_line:
        linestyle = 'None'
    else:
        linestyle = 'solid'
    # print("(plot_helpers.plot_outer_legend)linestyle:", linestyle)
    for idx, (label, line_data) in enumerate(zip(*for_plotlib)):
        labels.append(label)
        if idx > len(COLORS) - 1:
            color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        else:
            color = COLORS[idx]
        lines.append(
            plt.plot(
                line_data[0],
                line_data[1],
                'o-',
                color=color,
                label=label,
                ls=linestyle,
            )[0]
        )
    # print("(plot_helpers.plot_outer_legend)labels:", labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    scale_kwargs = dict()
    if xscale == 'symlog':
        linthreshx = get_linthreshx(
            for_plotlib[1],
        )
        scale_kwargs['linthreshx'] = linthreshx
    plt.xscale(xscale, **scale_kwargs)
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
        else:
            print("WARNING: %s directory doe not exist" % dir_)
    # print("(plot_helpers.parse_eval_dir)filtered:", filtered)
    return filtered


def create_plot_hp_layout(plot_dir, hp_plot_order, changing_hp):
    file_with_hp_layout_description = os.path.join(plot_dir, 'plot_hp_layout.txt')
    num_of_hps = len(hp_plot_order)
    if num_of_hps > 2:
        tmpl = '%s ' * (num_of_hps - 3) + '%s'
    else:
        tmpl = ''
    if num_of_hps > 1:
        line_hp_name = hp_plot_order[-2]
    else:
        line_hp_name = ''
    with open(file_with_hp_layout_description, 'w') as f:
        f.write('fixed hyper parameters: ' + tmpl % tuple(hp_plot_order[:-2]) + '\n')

        f.write('line hyper parameter: ' + line_hp_name + '\n')
        f.write('changing hyper parameter: ' + changing_hp)
    return tmpl


def get_y_specs(res_type, plot_parameter_names, metric_scales):
    if res_type in plot_parameter_names:
        ylabel = plot_parameter_names[res_type]
    else:
        ylabel = res_type
    if res_type in metric_scales:
        yscale = metric_scales[res_type]
    else:
        yscale = 'linear'
    return ylabel, yscale


def launch_plotting(data, line_label_format, fixed_hp_tmpl, path, xlabel, ylabel, xscale, yscale, no_line):
    on_descriptions = dict()
    for fixed_hps_tuple, plot_data in data.items():
        plot_data_on_labels = dict()
        for line_hp_value, line_data in plot_data.items():
            label = line_label_format.format(line_hp_value)
            if label in plot_data_on_labels:
                print("WARNING: specified formatting does not allow to distinguish '%s' in legend\n"
                      "fixed_hps_tuple: %s\n"
                      "falling to string formatting" % (line_hp_value, fixed_hps_tuple))
                label = '%s' % line_hp_value
                if label in plot_data_on_labels:
                    raise BadFormattingError(
                        line_label_format,
                        line_hp_value,
                        "Specified formatting does not allow to distinguish '%s' in legend\n"
                        "fixed_hps_tuple: %s\n"
                        "String formatting failed to fix the problem" % (line_hp_value, fixed_hps_tuple)
                    )
            plot_data_on_labels[line_label_format.format(line_hp_value)] = line_data
        on_descriptions[fixed_hp_tmpl % fixed_hps_tuple] = plot_data_on_labels
        # print("(plot_helpers.plot_hp_search)plot_data:", plot_data)

    counter = 0
    for description, plot_data in on_descriptions.items():
        file_name_without_ext = os.path.join(path, str(counter))
        plot_outer_legend(
            plot_data, description, xlabel, ylabel, xscale, yscale, file_name_without_ext, no_line
        )
        counter += 1


def plot_hp_search_optimizer(
        eval_dir,
        plot_dir,
        hp_plot_order,
        hp_names_file,
        metric_scales,
        xscale,
        no_line,
        line_label_format,
):
    plot_parameter_names = get_parameter_names(hp_names_file)
    changing_hp = hp_plot_order[-1]
    create_path(plot_dir)
    for_plotting = get_optimizer_evaluation_results(eval_dir, hp_plot_order, AVERAGING_NUMBER)
    pupil_names = sorted(list(for_plotting.keys()))
    result_types = sorted(list(for_plotting[pupil_names[0]].keys()))
    regimes = sorted(list(for_plotting[pupil_names[0]][result_types[0]].keys()))
    fixed_hp_tmpl = create_plot_hp_layout(plot_dir, hp_plot_order, changing_hp)
    # print("(plot_hp_search)plot_parameter_names:", plot_parameter_names)
    xlabel = plot_parameter_names[changing_hp]

    for pupil_name in pupil_names:
        for res_type in result_types:
            ylabel, yscale = get_y_specs(res_type, plot_parameter_names, metric_scales)
            for regime in sorted(regimes):
                path = os.path.join(plot_dir, pupil_name, res_type, regime)
                create_path(path)
                data = for_plotting[pupil_name][res_type][regime]
                launch_plotting(data, line_label_format, fixed_hp_tmpl, path, xlabel, ylabel, xscale, yscale, no_line)


def plot_hp_search_pupil(
        eval_dir,
        plot_dir,
        hp_plot_order,
        hp_names_file,
        metric_scales,
        xscale,
        no_line,
        line_label_format,
):
    plot_parameter_names = get_parameter_names(hp_names_file)
    changing_hp = hp_plot_order[-1]
    create_path(plot_dir)
    for_plotting = get_pupil_evaluation_results(eval_dir, hp_plot_order)
    dataset_names = sorted(list(for_plotting.keys()))
    result_types = sorted(list(for_plotting[dataset_names[0]].keys()))
    fixed_hp_tmpl = create_plot_hp_layout(plot_dir, hp_plot_order, changing_hp)
    xlabel = plot_parameter_names[changing_hp]
    for dataset_name in dataset_names:
        for res_type in result_types:
            ylabel, yscale = get_y_specs(res_type, plot_parameter_names, metric_scales)
            path = os.path.join(plot_dir, dataset_name, res_type)
            create_path(path)
            data = for_plotting[dataset_name][res_type]
            launch_plotting(data, line_label_format, fixed_hp_tmpl, path, xlabel, ylabel, xscale, yscale, no_line)
