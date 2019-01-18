import os


def prepend_cephfs(path):
    first = path[0]
    path = os.path.expanduser(path)
    prepended = '/cephfs'
    if os.path.exists(prepended) and first == '~':
        path = os.path.join(prepended, path)
    return path
