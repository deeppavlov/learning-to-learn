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
    plot_hp_search_pupil
from learning_to_learn.useful_functions import MissingHPError, HeaderLineError, ExtraHPError, BadFormattingError, \
    parse_x_select, parse_line_select
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "hp_order",
    help="Order of hyper parameters. Hyper parameter names as in run config separated by commas without spaces"
)
parser.add_argument(
    "eval_dir",
    help="Path to evaluation directory containing experiment results. \nTo process several evaluation directories"
         " use following format '<path1>,<path2>,...<pathi>:<pathi+1>,..:..'.\nAll possible combinations of sets"
         " separated by colons (in specified order) will be processed. \nYou have to provide paths relative to "
         "script. Edge characters of <path> can't be '/'"
)
parser.add_argument(
    "-pd",
    "--plot_dir",
    help="Path to directory where plots are going to be saved",
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
# parser.add_argument(
#     "-nlfp",
#     "--num_lines_for_plot",
#     help="Number of lines per one plot. Default is all lines."
# )
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
    '-f',
    "--line_label_format",
    help="Format used to add labels to legend. Default is '{:.0e}'",
    default='{:.0e}'
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
    help="select line hyper parameter values. Format: '<value1>,<value2>,...'. Default is None",
    default=None,
)
args = parser.parse_args()

AVERAGING_NUMBER = 3

select = dict()
if args.x_select is None:
    select['x_select'] = None
else:
    select['x_select'] = parse_x_select(args.x_select)
if args.line_select is None:
    select['line_select'] = None
else:
    select['line_select'] = parse_line_select(args.line_select)

style = dict(
    no_line=args.no_line,
    error=args.error,
    marker=args.marker,
)

eval_dirs = parse_eval_dir(args.eval_dir)
for eval_dir in eval_dirs:
    print(eval_dir)
    plot_dir = os.path.join(*list(os.path.split(eval_dir)[:-1]) + [args.plot_dir])
    hp_plot_order = args.hp_order.split(',')

    metric_scales = dict()
    if args.metric_scales is not None:
        for one_metric_scale in args.metric_scales.split(','):
            [metric, scale] = one_metric_scale.split(':')
            metric_scales[metric] = scale

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    plot_parameter_names = get_parameter_names(args.hp_names_file)
    xscale = args.xscale

    if args.model == 'optimizer':
        try:
            plot_hp_search_optimizer(
                eval_dir,
                plot_dir,
                hp_plot_order,
                args.hp_names_file,
                metric_scales,
                args.xscale,
                style,
                args.line_label_format,
                select,
            )
        except MissingHPError as e:
            print("WARNING: can not plot results in '%s' because they miss hyper parameter %s.\n"
                  "Try other hyper parameter configuration\n" % (eval_dir, e.hp_name))
        except ExtraHPError as e:
            print(
                "WARNING: can not plot results in '%s' because not all hyper parameters were provided\n"
                "hp_order: %s\n"
                "hp_save_order(in hp_layout.txt): %s" % (eval_dir, e.hp_order, e.hp_names)
            )
        except BadFormattingError as e:
            print(
                "WARNING: can not plot results in '%s' because "
                "formatting does not allow put labels in legend\n" % eval_dir,
                e.message
            )
        except:
            raise
    elif args.model == 'pupil':
        try:
            plot_hp_search_pupil(
                eval_dir,
                plot_dir,
                hp_plot_order,
                args.hp_names_file,
                metric_scales,
                args.xscale,
                style,
                args.line_label_format,
                select,
            )
        except HeaderLineError as e:
            print(
                "WARNING: can not plot results in '%s' because header line '%s' is invalid" % (eval_dir, e.header_line)
            )
        except ExtraHPError as e:
            print(
                "WARNING: can not plot results in '%s' because not all hyper parameters were provided\n"
                "hp_order: %s\n"
                "hp_names(in file): %s" % (eval_dir, e.hp_order, e.hp_names)
            )
        except MissingHPError as e:
            print("WARNING: can not plot results in '%s' because they miss hyper parameter %s.\n"
                  "Try other hyper parameter configuration\n" % (eval_dir, e.hp_name))
        except BadFormattingError as e:
            print(
                "WARNING: can not plot results in '%s' because "
                "formatting does not allow put labels in legend\n" % eval_dir,
                e.message
            )
        except:
            raise

