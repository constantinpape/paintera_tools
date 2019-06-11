# paintera_tools

Tools for genetrating curating and merging paintera datasets

## Functionality

This package provides functionality to interacte with paintera datasets.
The three main applications are:
- convert n5 or h5 container to paintera dataset via `convert_to_paintera_format`
- split objects via `interactive_splitter` or `batch_splitter`
- export merge segmentation  via `serialize_from_commit`

## Installation

This package relies on [cluster_tools](https://github.com/constantinpape/cluster_tools).
It should be used from a conda environment with cluster tools and all its dependencies.
I will try to push both cluster_tools and paintera_tools to conda-forge soon.

## Usage

For usage examples see `examples/`. You can obtain the example [cremi](https://cremi.org/) data from [here](https://drive.google.com/open?id=1E6j77gV0iwquSxd7KmmuXghgFcyuP7WW).
