# paintera_tools

Tools for genetrating, curating and merging [paintera](https://github.com/saalfeldlab/paintera) datasets.

## Functionality

This package provides functionality to interacte with paintera datasets.
The three main applications are:
- convert n5 or h5 container to paintera dataset via `convert_to_paintera_format`
- split objects via `interactive_splitter` or `batch_splitter`
- export merge segmentation  via `serialize_from_commit`

## Installation

You can install this package via conda (only python 3.7 and linux for now):
```
conda install -c conda-forge -c cpape paintera_tools
```

To set up a developement environment, you will need [cluster_tools](https://github.com/constantinpape/cluster_tools)
and its dependencies.

## Usage

For usage examples see `examples/`. You can obtain the example [cremi](https://cremi.org/) data from [here](https://drive.google.com/file/d/1E_Wpw9u8E4foYKk7wvx5RPSWvg_NCN7U/view?usp=sharing).
