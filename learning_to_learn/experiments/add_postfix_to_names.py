import sys
import os

postfix = sys.argv[1]
directory = sys.argv[2]
names = sys.argv[3:]
full_names = [os.path.join(directory, name) for name in names]

for name in full_names:
    if os.path.isdir(name):
        os.rename(name, name + postfix)
    elif os.path.isfile(name):
        path = os.path.split(name)
        dir = path[:-1]
        file_name = path[-1]
        if '.' in file_name:
            tmp = file_name.split('.')
            base = '.'.join(tmp[:-1])
            ext = tmp[-1]
            new_file_name = '.'.join([base + postfix, ext])
        else:
            new_file_name = file_name + postfix
        new_name = os.path.join(*dir, new_file_name)
        os.rename(name, new_name)
    else:
        print("WARNING: file or directory %s does not exist" % name)
