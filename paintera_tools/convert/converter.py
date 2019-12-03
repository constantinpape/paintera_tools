import os
import json
import luigi
from cluster_tools.paintera import ConversionWorkflow
from cluster_tools.downscaling import DownscalingWorkflow

from ..util import write_global_config


def convert_to_paintera_format(path, raw_key, in_key, out_key,
                               label_scale, resolution,
                               tmp_folder, target, max_jobs, max_threads,
                               assignment_path='', assignment_key='',
                               label_block_mapping_compression='gzip',
                               copy_labels=False, convert_to_label_multisets=False,
                               restrict_sets=None, restrict_scales=None):
    """ Convert n5 label dataset to n5 paintera format.

    Produces output group with the subgroups:
    - data: pyramid labels data
    - unique-labels: unique labels per chunks
    - label-to-block-mapping: mapping of label-ids to block-ids
    - fragment-segment-assignment: optional fragment to segment assignment
      (only if assignments are given).
    See description at https://github.com/saalfeldlab/paintera#paintera-data-format.

    Arguments:
        path [str] - path to n5 container with raw and label data.
        raw_key [str] - name of the raw multi-scale pyramid group.
        in_key [str] - name of the labels dataset.
        out_key [str] - name of the output paintera group.
        label_scale [int] - scale level of the  labels compared to raw data.
        resolution [tuple] - base resolution of the raw data.
        tmp_folder [str] - folder to store tempory data.
        target [str] - computation target, can be 'local', 'slurm' or 'lsf'.
        max_jobs [int] - maximum number of jobs used in computation.
        max_threads [int] - maximum  number of threads used in computation.
        assignment_path [str] - optional path to  assignments for fragment-segment-assignments (default: '')
        assignment_key [str] - optional key to assignments (default: '')
        label_block_mapping_compression [str] - compression used for label-to-block-mapping.
            If the default ('gzip') leads to issues (can happen for too small block) use
            'raw' instead. (default: 'gzip')
            copy_labels [bool] - copy the dataset with input labels instead of making a soft-link (default: False)
        convert_to_label_multisets [bool] - convert to label multiset instead of label arrays. (default: False)
        restrict_sets [listlike] - label multiset restrictions for all scales.
            Must be given if convert_to_label_multisets is set. (default: None)
        restrict_scales [int] - restrict the number of scales for downsampling. (default: None)
    """
    if convert_to_label_multisets:
        assert restrict_sets is not None

    config_folder = os.path.join(tmp_folder, 'configs')
    write_global_config(config_folder)

    configs = ConversionWorkflow.get_config()
    config = configs['downscaling']
    config.update({"library_kwargs": {"order": 0}, "mem_limit": 12, "time_limit": 120})
    with open(os.path.join(config_folder, "downscaling.config"), 'w') as f:
        json.dump(config, f)

    block_mapping_conf = configs['label_block_mapping']
    block_mapping_conf.update(
        {
            'mem_limit': 100,
            'time_limit': 360,
            'threads_per_job': max_threads,
            'compression': label_block_mapping_compression
        }
    )
    with open(os.path.join(config_folder, 'label_block_mapping.config'), 'w') as f:
        json.dump(block_mapping_conf, f)

    if convert_to_label_multisets:
        create_conf = configs['create_multiset']
        create_conf.update({'time_limit': 240, 'mem_limit': 4})
        with open(os.path.join(config_folder, 'create_multiset.config'), 'w') as f:
            json.dump(create_conf, f)

        ds_conf = configs['downscale_multiset']
        ds_conf.update({'time_limit': 360, 'mem_limit': 8})
        with open(os.path.join(config_folder, 'downscale_multiset.config'), 'w') as f:
            json.dump(ds_conf, f)

    task = ConversionWorkflow(tmp_folder=tmp_folder, config_dir=config_folder,
                              max_jobs=max_jobs, target=target,
                              path=path, raw_key=raw_key,
                              label_in_key=in_key, label_out_key=out_key,
                              assignment_path=assignment_path, assignment_key=assignment_key,
                              label_scale=label_scale, resolution=resolution,
                              use_label_multiset=convert_to_label_multisets,
                              restrict_sets=restrict_sets, copy_labels=copy_labels)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Conversion to paintera format failed"


def downscale(path, input_key, output_key,
              scale_factors, halos,
              tmp_folder, target, max_jobs, resolution=None,
              library='skimage', **library_kwargs):
    """ Downscale input data to obtain pyramid.

    Arguments:
        path [str] - path to n5 container with input data and for output data.
        input_key [str] - name of input data.
        output_key [str] - name of output multi-scale group.
        scale_factors [listlike] - factors used for downsampling.
        halos [listlike] - halo values used for downsampling.
        tmp_folder [str] - folder to store tempory data.
        target [str] - computation target, can be 'local', 'slurm' or 'lsf'.
        max_jobs [int] - maximum number of jobs used in computation.
        resolution [tuple] - resolution of the input data. (default: None)
        library [str] - library used for down-sampling (default: 'skimage')
        library_kwargs [kwargs] - keyword arguments for the down-scaling function.
    """
    assert len(scale_factors) == len(halos)

    config_folder = os.path.join(tmp_folder, 'configs')
    write_global_config(config_folder)

    task = DownscalingWorkflow
    configs = task.get_config()

    config = configs['downscaling']
    config.update({"mem_limit": 12, "time_limit": 120, "library": library})
    if library_kwargs:
        config.update({"library_kwargs": library_kwargs})
    with open(os.path.join(config_folder, "downscaling.config"), 'w') as f:
        json.dump(config, f)

    metadata = {}
    if resolution:
        metadata.update({'resolution': resolution})

    t = DownscalingWorkflow(tmp_folder=tmp_folder, config_dir=config_folder,
                            max_jobs=max_jobs, target=target,
                            input_path=path, input_key=input_key,
                            output_key_prefix=output_key,
                            scale_factors=scale_factors, halos=halos,
                            metadata_format='paintera', metadata_dict=metadata)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Conversion to paintera format failed"
