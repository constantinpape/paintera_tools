import os
import json
import numpy as np

import luigi
import vigra
import z5py
from cluster_tools.write import WriteLocal, WriteSlurm
from cluster_tools.copy_volume import CopyVolumeLocal, CopyVolumeSlurm

from ..util import save_assignments, make_dense_assignments, find_uniques, write_global_config


# TODO wrap this in a luigi.Task
def serialize_assignments(g, ass_key,
                          save_path, unique_key, save_key,
                          locked_segments=None, relabel_output=False,
                          map_to_background=None):

    # load the unique ids
    f = z5py.File(save_path)
    fragment_ids = f[unique_key][:]

    # load the assignments
    assignments = g[ass_key][:].T
    dense_assignments = make_dense_assignments(fragment_ids, assignments)

    # only keep assignments corresponding to locked segments
    # if locked segments are given
    if locked_segments is not None:
        locked_mask = np.in1d(dense_assignments[:, 1], locked_segments)
        dense_assignments[:, 1][np.logical_not(locked_mask)] = 0

    # relabel the assignments consecutively
    if relabel_output:
        values = dense_assignments[:, 1]
        vigra.analysis.relabelConsecutive(values, start_label=1, keep_zeros=True,
                                          out=values)
        dense_assignments[:, 1] = values

    if map_to_background:
        bg_mask = np.isin(dense_assignments[:, 1], map_to_background)
        dense_assignments[:, 1][bg_mask] = 0

    save_assignments(dense_assignments, save_path, save_key)


def serialize_merged_segmentation(path, key, out_path, out_key, ass_path, ass_key,
                                  tmp_folder, max_jobs, target):
    task = WriteLocal if target == 'local' else WriteSlurm
    config_folder = os.path.join(tmp_folder, 'configs')
    config = task.default_task_config()

    block_shape = z5py.File(path, 'r')[key].chunks
    config.update({'chunks': block_shape, 'allow_empty_assignments': True})
    with open(os.path.join(config_folder, 'write.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_folder, max_jobs=max_jobs,
             input_path=path, input_key=key,
             output_path=out_path, output_key=out_key,
             assignment_path=ass_path, assignment_key=ass_key,
             identifier='merge-paintera-seg')
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Writing merged segmentation failed")


def copy_segmentation(path, in_key, out_path, out_key,
                      tmp_folder, max_jobs, target):
    task = CopyVolumeLocal if target == 'local' else CopyVolumeSlurm
    config_folder = os.path.join(tmp_folder, 'configs')

    t = task(tmp_folder=tmp_folder, config_dir=config_folder, max_jobs=max_jobs,
             input_path=path, input_key=in_key, output_path=out_path, output_key=out_key,
             prefix='copy_serialize')

    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Copying segmentation failed"


def serialize_with_assignments(path, g, out_path, out_key,
                               seg_in_key, tmp_folder,
                               max_jobs, target,
                               locked_segments, relabel_output,
                               map_to_background):
    assignment_in_key = 'fragment-segment-assignment'
    config_folder = os.path.join(tmp_folder, 'configs')

    save_path = os.path.join(tmp_folder, 'assignments.n5')
    unique_key = 'uniques'
    assignment_key = 'assignments'

    # 1.) find the unique ids in the base segemntation
    find_uniques(path, seg_in_key, save_path, unique_key,
                 tmp_folder, config_folder, max_jobs, target)

    # 2.) make and serialize new assignments
    print("Serializing assignments ...")
    serialize_assignments(g, assignment_in_key,
                          save_path, unique_key, assignment_key,
                          locked_segments, relabel_output,
                          map_to_background)

    # 3.) write the new segmentation
    print("Serializing new segmentation ...")
    serialize_merged_segmentation(path, seg_in_key,
                                  out_path, out_key,
                                  save_path, assignment_key,
                                  tmp_folder, max_jobs, target)


def serialize_from_commit(path, key, out_path, out_key,
                          tmp_folder, max_jobs, target, scale=0,
                          locked_segments=None, relabel_output=False,
                          map_to_background=None):
    """ Serialize corrected segmentation from commited project.
    """
    f = z5py.File(path, 'r')
    g = f[key]

    os.makedirs(tmp_folder, exist_ok=True)
    config_folder = os.path.join(tmp_folder, 'configs')

    # make sure this is a paintera group
    seg_key = 'data'
    assignment_in_key = 'fragment-segment-assignment'
    assert seg_key in g
    have_assignments = assignment_in_key in g

    # prepare cluster tools tasks
    seg_in_key = os.path.join(key, seg_key, 's%i' % scale)
    block_shape = f[seg_in_key].chunks
    write_global_config(config_folder, block_shape)

    if have_assignments:
        serialize_with_assignments(path, g, out_path, out_key, seg_in_key,
                                   tmp_folder, max_jobs, target, locked_segments,
                                   relabel_output, map_to_background=map_to_background)
    else:
        copy_segmentation(path, seg_in_key, out_path, out_key,
                          tmp_folder, max_jobs, target)
