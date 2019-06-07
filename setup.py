from distutils.core import setup
from pkgutil import walk_packages

import paintera_tools
from paintera_tools import __version__


def find_packages(path, prefix):
    yield prefix
    prefix = prefix + "."
    for _, name, ispkg in walk_packages(path, prefix):
        if ispkg:
            yield name


setup(name='paintera_tools',
      version=__version__,
      description='Tools for generating, curating and merging paintera datasets',
      author='Constantin Pape',
      packages=list(find_packages(paintera_tools.__path__, paintera_tools.__name__)))
