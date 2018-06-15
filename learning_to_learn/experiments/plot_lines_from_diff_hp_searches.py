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

from learning_to_learn.experiments.plot_helpers import get_parameter_names, plot_hp_search_optimizer, parse_eval_dir, \
    plot_hp_search_pupil, fixed_hps_from_str, parse_metric_scales_str
from learning_to_learn.useful_functions import MissingHPError, HeaderLineError, ExtraHPError, BadFormattingError, \
    parse_x_select, parse_line_select, broadcast_list, broadcast_many_lists, split_strings_by_char, convert
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "hp_orders",
    help="Order of hyper parameters. Hyper parameter names as in run config separated by commas without spaces."
         " You have to provide hyper parameters for all evaluation directories separating them with colon character"
)
parser.add_argument(
    "eval_dir",
    help="Path to evaluation directory containing experiment results. \nTo process several evaluation directories"
         " use following format '<path1>,<path2>,...<pathi>:<pathi+1>,..:..'.\nAll possible combinations of sets"
         " separated by colons (in specified order) will be processed. \nYou have to provide paths relative to "
         "script. Edge characters of <path> can't be '/'"
)
parser.add_argument(
    '-lbl',
    "--labels",
    help="labels for plotted lines. Labels are provided in format '<label1>,<label2>,...:<labeli>...'."
         " Labels for different evaluation directories are separated by colons. It is obligatory parameter'",
    default=None
)
parser.add_argument(
    "-pd",
    "--plot_dir",
    help="Path to directory where plots are going to be saved. In contrast to 'plot_hp_search.py' it is full path.",
    default='plots',
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
parser.add_argument(
    '-m',
    "--model",
    help="Optimized model type. It can be 'pupil' or optimizer or 'optimizer'. Default is 'optimizer'",
    default='optimizer',
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
    '-lst',
    '--line_select',
    help="select line hyper parameter values. Format: '<value1>,<value2>,...:<valuei>...'. Line hyper parameter"
         " values for different evaluation directories are separated by colons."
         "Default is None",
    default=None,
)
parser.add_argument(
    '-rg',
    "--regimes",
    help="select regimes from which lines are taken. Format: '<regime1>,<regime2>,...:<regimei>...'. "
         " Regimes for different evaluation directories are separated by colons. This option is used only for "
         "optimizers. Default is validation",
    default='validation',
)
parser.add_argument(
    '-pn',
    "--pupil_names",
    help="select pupil names for which lines are taken. Format: '<pupil1>,<pupil2>,...:<pupili>...'. "
         " Pupil names for different evaluation directories are separated by colons. This option is used only for "
         "optimizers. Default is 'pretrain0",
    default='pretrain0',
)
parser.add_argument(
    '-dn',
    "--dataset_names",
    help="select dataset names for which lines are taken. Format: '<dataset_name1>,<dataset_name2>,...:"
         "<dataset_namei>...'. "
         " Dataset_names for different evaluation directories are separated by colons. This option is used only for "
         "optimizers. Default is 'valid'",
    default='valid',
)
parser.add_argument(
    '-fhp',
    "--fixed_hp_values",
    help="Values of fixed hyper parameters. Format: '(<hp1>,<hp2>...),()...:()...'",
    default=None
)

args = parser.parse_args()

eval_dirs = parse_eval_dir(args.eval_dir)

[lines_by_ed, labels_by_ed, fixed_hp_by_ed,
 regimes_by_ed, pupil_names_by_ed, dataset_names_by_ed] = split_strings_by_char(
    [args.lines, args.labels, args.fixed_hp_values, args.regimes, args.pupil_names, args.dataset_names], ':'
)
num_lines = [len(ed_lines.split(',')) for ed_lines in lines_by_ed]
num_ed = len(eval_dirs)
fixed_hp_by_ed = broadcast_list(fixed_hp_by_ed, num_ed)

if args.model == 'pupil':
    dataset_names_by_ed = broadcast_list(dataset_names_by_ed, num_ed)

elif args.model == 'optimizer':
    [regimes_by_ed, pupil_names_by_ed] = broadcast_many_lists(
        [regimes_by_ed, pupil_names_by_ed],
        num_ed
    )


line_retrieve_inf = dict()
for eval_dir, ed_lines, ed_fixed_hps, ed_regimes, ed_pupil_names, ed_dataset_names, ed_labels, nlines in \
        zip(
            eval_dirs, lines_by_ed, fixed_hp_by_ed, regimes_by_ed,
            pupil_names_by_ed, dataset_names_by_ed, labels_by_ed, num_lines
        ):
    [ed_lines, ed_fixed_hps, ed_regimes, ed_pupil_names, ed_dataset_names, ed_labels] = split_strings_by_char(
        [ed_lines, ed_fixed_hps, ed_regimes, ed_pupil_names, ed_dataset_names, ed_labels], ','
    )

    [ed_regimes, ed_pupil_names, ed_dataset_names, ed_fixed_hps] = broadcast_many_lists(
        [ed_regimes, ed_pupil_names, ed_dataset_names, ed_fixed_hps],
        nlines,
    )

    ed_lines = [convert(x, 'float') for x in ed_lines]
    ed_fixed_hps = [fixed_hps_from_str(x) for x in ed_fixed_hps]

    line_retr_inf = list()
    if args.model == 'pupil':
        for line_hp, fixed_hps, dataset_name, label in zip(
                ed_lines, ed_fixed_hps, ed_dataset_names, ed_labels
        ):
            line_retr_inf.append(
                dict(
                    line_hp=line_hp,
                    fixed_hps=fixed_hps,
                    dataset_name=dataset_name,
                    label=label,
                )
            )
    elif args.model == 'optimizer':
        for line_hp, fixed_hps, regime, pupil_name, label in zip(
                ed_lines, ed_fixed_hps, ed_regimes, ed_pupil_names, ed_labels
        ):
            line_retr_inf.append(
                dict(
                    line_hp=line_hp,
                    fixed_hps=fixed_hps,
                    regime=regime,
                    pupil_name=pupil_name,
                    label=label,
                )
            )

if args.x_select is None:
    x_select = None
else:
    x_select = parse_x_select(args.x_select)

style = dict(
    no_line=args.no_line,
    error=args.error,
    marker=args.marker,
)

metric_scales = parse_metric_scales_str(args.metric_scales)

hp_plot_order = args.hp_order.split(',')
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

plot_parameter_names = get_parameter_names(args.hp_names_file)
xscale = args.xscale





    

