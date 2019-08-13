import runpy
from setuptools import setup, find_packages

version = runpy.run_path('paintera_tools/version.py')['__version__']
setup(name='paintera_tools',
      packages=find_packages(exclude=['test']),
      version=version,
      description='Tools for generating, curating and merging paintera datasets',
      author='Constantin Pape',
      url='https://github.com/constantinpape/paintera_tools',
      license='MIT')
