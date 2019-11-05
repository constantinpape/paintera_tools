import os
import json
import numpy as np

import luigi
import vigra
import nifty.tools as nt
from elf.io import open_file
from cluster_tools.write import WriteLocal, WriteSlurm
from cluster_tools.copy_volume import CopyVolumeLocal, CopyVolumeSlurm

from ..util import save_assignments, make_dense_assignments, find_uniques, write_global_config


class SerializeAssignments(luigi.Task):
    tmp_folder = luigi.Parameter()
    assignment_path = luigi.Parameter()
    assignment_key = luigi.Parameter()
    unique_path = luigi.Parameter()
    unique_key = luigi.Parameter()
    out_path = luigi.Parameter()
    out_key = luigi.Parameter()
    locked_segments = luigi.ListParameter(default=None)
    relabel_output = luigi.BoolParameter(default=False)
    map_to_background = luigi.ListParameter(default=None)

    def run(self):
        # load the unique fragment ids
        with open_file(self.unique_path, 'r') as f:
            ds = f[self.unique_key]
            fragment_ids = ds[:]

        # load the assignments and make them dense
        with open_file(self.assignment_path, 'r') as f:
            ds = f[self.assignment_key]
            assignments = ds[:].T
        dense_assignments = make_dense_assignments(fragment_ids, assignments)

        # if locked segments are given,
        # only keep assignments corresponding to locked segments
        if self.locked_segments is not None:
            locked_mask = np.in1d(dense_assignments[:, 1], self.locked_segments)
            dense_assignments[:, 1][np.logical_not(locked_mask)] = 0

        # relabel the assignments consecutively if specified
        if self.relabel_output:
            values = dense_assignments[:, 1]
            vigra.analysis.relabelConsecutive(values, start_label=1, keep_zeros=True,
                                              out=values)
            dense_assignments[:, 1] = values

        if self.map_to_background is not None:
            bg_mask = np.isin(dense_assignments[:, 1], self.map_to_background)
            dense_assignments[:, 1][bg_mask] = 0

        save_assignments(dense_assignments, self.out_path, self.out_key)
        log_path = self.output().path
        with open(log_path, 'w') as f:
            f.write("serialized paintera assignments")

    def output(self):
        log_path = os.path.join(self.tmp_folder, 'serialize_assignments.log')
        return luigi.LocalTarget(log_path)


def serialize_assignments(tmp_folder,
                          assignment_path, assignment_key,
                          unique_path, unique_key,
                          out_path, out_key,
                          locked_segments=None, relabel_output=False,
                          map_to_background=None):
    task = SerializeAssignments
    t = task(tmp_folder=tmp_folder,
             assignment_path=assignment_path, assignment_key=assignment_key,
             unique_path=unique_path, unique_key=unique_key,
             out_path=out_path, out_key=out_key,
             locked_segments=locked_segments, relabel_output=relabel_output,
             map_to_background=map_to_background)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Serializing the assignments failed"


def serialize_merged_segmentation(path, key, out_path, out_key, ass_path, ass_key,
                                  tmp_folder, max_jobs, target):
    task = WriteLocal if target == 'local' else WriteSlurm
    config_folder = os.path.join(tmp_folder, 'configs')
    config = task.default_task_config()

    block_shape = open_file(path, 'r')[key].chunks
    config.update({'chunks': block_shape, 'allow_empty_assignments': True,
                   'time_limit': 180})
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


def serialize_with_assignments(path, paintera_key, out_path, out_key,
                               seg_in_key, tmp_folder,
                               max_jobs, target,
                               locked_segments, relabel_output,
                               map_to_background):
    assignment_in_key = os.path.join(paintera_key, 'fragment-segment-assignment')
    config_folder = os.path.join(tmp_folder, 'configs')

    save_path = os.path.join(tmp_folder, 'assignments.n5')
    unique_key = 'uniques'
    assignment_key = 'assignments'

    # 1.) find the unique ids in the base segemntation
    find_uniques(path, seg_in_key, save_path, unique_key,
                 tmp_folder, config_folder, max_jobs, target)

    # 2.) make and serialize new assignments
    print("Serializing assignments ...")
    serialize_assignments(tmp_folder,
                          path, assignment_in_key,
                          save_path, unique_key,
                          save_path, assignment_key,
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
    f = open_file(path, 'r')
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
        serialize_with_assignments(path, key, out_path, out_key, seg_in_key,
                                   tmp_folder, max_jobs, target, locked_segments,
                                   relabel_output, map_to_background=map_to_background)
    else:
        copy_segmentation(path, seg_in_key, out_path, out_key,
                          tmp_folder, max_jobs, target)


def extract_from_commit(path, key, scale=0, relabel_output=False, n_threads=8):
    """ Extract corrected segmentation from commited project
    and return it as array.
    """
    f = open_file(path, 'r')
    g = f[key]

    # make sure this is a paintera group
    seg_key = 'data'
    assignment_in_key = 'fragment-segment-assignment'
    assert seg_key in g
    have_assignments = assignment_in_key in g
    seg_in_key = os.path.join(seg_key, 's%i' % scale)

    # TODO support label multiset here !
    ds = g[seg_in_key]
    ds.n_threads = n_threads
    seg = ds[:]

    if have_assignments:
        fragment_ids = np.unique(seg)
        assignments = g[assignment_in_key][:].T
        assignments = make_dense_assignments(fragment_ids, assignments)
        if relabel_output:
            assignments[:, 1] = vigra.analysis.relabelConsecutive(assignments[:, 1], start_label=1, keep_zeros=True)
        assignments = dict(zip(assignments[:, 0], assignments[:, 1]))
        seg = nt.takeDict(assignments, seg)

    elif relabel_output:
        seg = vigra.analysis.relabelConsecutive(seg, start_label=1, keep_zeros=True)

    return seg
