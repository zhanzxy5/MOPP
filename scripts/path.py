import os
from os.path import *


def abs_file_path(file, relative_path):
    return os.path.abspath(os.path.join(os.path.split(os.path.abspath(file))[0], relative_path))


__all__ = [_s for _s in dir() if not _s.startswith('_')]
__all__.extend(os.path.__all__)