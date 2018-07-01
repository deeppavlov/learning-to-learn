import sys
import os
import matplotlib.pyplot as plt
from matplotlib import rc

from pathlib import Path  # if you haven't already done so
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[2]
sys.path.append(str(root))
try:
    sys.path.remove(str(parent))
except ValueError:  # Already removed
    pass

from learning_to_learn.useful_functions import create_path, InvalidArgumentError

COLORS = ['r', 'g', 'b', 'k', 'c', 'magenta', 'brown', 'darkviolet', 'pink', 'yellow', 'gray', 'orange', 'olive']
DPI = 900
FORMATS = ['pdf', 'png']

FONT = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **FONT)

save_path = sys.argv[1]  # no extension!
create_path(save_path, file_name_is_in_path=True)
xlabel = sys.argv[2]
ylabel = sys.argv[3]
xscale = sys.argv[4]
yscale = sys.argv[5]
limit = sys.argv[6]
if limit == 'None':
    limit = None
else:
    limit = int(limit)
shift = sys.argv[7]
if limit == 'None':
    shift = None
else:
    shift = int(shift)
plot_data = sys.argv[8:]
if len(plot_data) % 2 != 0:
    raise InvalidArgumentError(
        "Every file has to be provided with line name (number of arguments has to be even)",
        [plot_data, len(plot_data)],
        'plot_data',
        "len(plot_data) % 2 == 0"
    )

line_names = [plot_data[2*i] for i in range(len(plot_data) // 2)]
file_names = [plot_data[2*i+1] for i in range(len(plot_data) // 2)]

# print(line_names)

data = list()
for line_name, file_name in zip(line_names, file_names):
    with open(file_name, 'r') as f:
        lines = f.read().split('\n')
    lines = lines[:limit]
    while len(lines) > 0 and len(lines[-1]) == 0:
        lines = lines[:-1]
    xlist = list()
    ylist = list()
    for line in lines:
        x, y = line.split()
        if shift is None:
            xlist.append(float(x))
        else:
            xlist.append(float(x) + shift)
        ylist.append(float(y))
    data.append((line_name, xlist, ylist))

for idx, (line_name, x, y) in enumerate(data):
    plt.plot(x, y, color=COLORS[idx], label=line_name)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xscale(xscale)
plt.yscale(yscale)
plt.legend(loc='lower center')

for format in FORMATS:
    if format == 'pdf':
        fig_path = os.path.join(save_path + '.pdf')
        r = plt.savefig(fig_path)
    if format == 'png':
        fig_path = os.path.join(save_path + '.png')
        r = plt.savefig(fig_path, dpi=DPI)