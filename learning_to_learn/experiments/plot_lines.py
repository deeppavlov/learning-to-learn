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
    plot_hp_search_pupil, parse_metric_scales_str, plot_outer_legend, get_parameter_name
from learning_to_learn.useful_functions import MissingHPError, HeaderLineError, ExtraHPError, BadFormattingError, \
    parse_x_select, parse_line_select, create_path, parse_path_comb, parse_1_line_dir, keys_from_list_in_dict, \
    extract_line_from_file, select_by_x, nested2string
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "dirs",
    help="directories in which results lay"
)
parser.add_argument(
    'labels',
    help="Labels for lines in corresponding directories"
)
parser.add_argument(
    'xshift',
    help="Shift of values of horizontal parameters. If horizontal axis used for step than step values from collected "
         "data are likely start from zero whereas in fact it is better for understanding when they start from 1."
         " It is also useful if log scale is used."
)

parser.add_argument(
    "-pn",
    "--plot_name",
    help="Path to file where plots are going to be saved. Given relative to experiments directory. DON'T ADD"
         "AN EXTENSION",
    default='line',
)
parser.add_argument(
    "-xs",
    "--xscale",
    help="x axis scale. It can be log or linear. Default is linear",
    default='linear',
)
parser.add_argument(
    "-ms",
    "--metric_scales",
    help="Scales for metrics. Available metrics are 'accuracy', 'bpc', 'loss', 'perplexity'. "
         "Scales are provided in following format <metric>:<scale>,<metric>:<scale>. "
         "Default is linear"
)
parser.add_argument(
    '-ds',
    '--datasets',
    help='list of datasets plot'
)
parser.add_argument(
    '-xl',
    '--xlabel',
)
parser.add_argument(
    '-e',
    "--error",
    help="Error style. If 'None' no error bars are plotted. Possible options 'fill', 'bar'",
    default='None',
)
parser.add_argument(
    '-mk',
    "--marker",
    help="Marker style. default is o",
    default='o',
)
parser.add_argument(
    '-xst',
    '--x_select',
    help="select x values from specified range. Use following format '[x1,x2][x3,x4]...'. Default is None",
    default=None,
)
parser.add_argument(
    "-nl",
    "--no_line",
    help="Do not link dots with line. Default is True",
    action='store_true'
)
parser.add_argument(
    '-hpnf',
    "--hp_names_file",
    help="File with hyper parameter names. All available files are in the same directory with this script",
    default='hp_plot_names_english.conf'
)
args = parser.parse_args()
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
dirs = parse_path_comb(args.dirs)

labels = args.labels.split(',')
datasets = args.datasets.split(',')

lines_by_metrics = dict()

label_idx = 0

if args.x_select is None:
    x_select = None
else:
    x_select = parse_x_select(args.x_select)
for label, dir in zip(labels, dirs):
    dir_contents = parse_1_line_dir(dir)
    for metric, metric_data in dir_contents.items():
        if metric not in lines_by_metrics:
            lines_by_metrics[metric] = dict()
        datasets_present = keys_from_list_in_dict(metric_data, datasets)
        if len(datasets_present) > 1:
            for dataset in datasets_present:
                lines_by_metrics[metric][label + '_' + dataset] = select_by_x(
                    extract_line_from_file(
                        metric_data[dataset]
                    ),
                    x_select
                )
        else:
            lines_by_metrics[metric][label] = select_by_x(
                extract_line_from_file(
                    metric_data[datasets_present[0]]
                ),
                x_select
            )
style = dict(
    no_line=args.no_line,
    error=args.error,
    marker=args.marker,
)
plot_parameter_names = get_parameter_names(args.hp_names_file)
xscale = args.xscale
metric_scales = parse_metric_scales_str(args.metric_scales)

create_path(args.plot_name, file_name_is_in_path=True)
for metric, lines_for_metric in lines_by_metrics.items():
    file_name = args.plot_name + '_' + metric
    ylabel = get_parameter_name(plot_parameter_names, metric)
    if metric in metric_scales:
        yscale = metric_scales[metric]
    else:
        yscale = 'linear'
    xlabel = args.xlabel
    description = nested2string(lines_for_metric)
    plot_outer_legend(
        lines_for_metric,
        description,
        xlabel,
        ylabel,
        xscale,
        yscale,
        file_name,
        style,
        shifts=[int(args.xshift), 0]
    )

