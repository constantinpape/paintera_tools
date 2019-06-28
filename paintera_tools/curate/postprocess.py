import os
import luigi
import z5py
import numpy as np

from cluster_tools.postprocess import SizeFilterAndGraphWatershedWorkflow, ConnectedComponentsWorkflow
from ..util import compute_graph_and_weights
from ..serialize import serialize_from_commit
from .splitter import assignment_saver


def make_graph_assignments(f, node_ids, assignments, out_key, n_threads):
    # make node labels from assignments
    assignment_dict = dict(zip(assignments[:, 0],
                               assignments[:, 1]))
    # we need dense assignments for this to work
    max_id = int(node_ids.max())
    node_labels = np.array([assignment_dict.get(node_id, node_id) for node_id in range(max_id + 1)],
                           dtype='uint64')

    # save temporary node labels
    new_chunks = (min(100000, len(node_labels)),)
    ds = f.require_dataset(out_key, shape=node_labels.shape,
                           chunks=new_chunks, compression='gzip',
                           dtype='uint64')
    ds.n_threads = n_threads
    ds[:] = node_labels


def prepare_postprocesing(paintera_path, paintera_key,
                          boundary_path, boundary_key,
                          data_key, assignment_key,
                          assignment_out_key, backup_assignments,
                          tmp_folder, target, max_jobs, n_threads):

    data_key = os.path.join(paintera_key, data_key)
    assignment_key = os.path.join(paintera_key, assignment_key)

    f = z5py.File(paintera_path)
    assignments = f[assignment_key][:].T
    chunks = f[assignment_key].chunks
    # make backup of assignments
    if backup_assignments:
        bkp_key = os.path.join(paintera_key, 'assignments-bkp')
        print("Making back-up @", paintera_path, ":", bkp_key)
        assignment_saver(paintera_path, bkp_key, n_threads,
                         assignments, chunks)

    exp_path = ''
    f = z5py.File(exp_path)

    # load the graph node ids (= watershed / fragment ids)
    ds_nodes = f['s0/graph/nodes']
    ds_nodes.n_threads = n_threads
    node_ids = ds_nodes[:]

    # save ass
    if assignment_out_key not in f:
        make_graph_assignments(f, node_ids, assignments, assignment_out_key,
                               n_threads)
    return node_ids, exp_path, chunks


def connected_components_and_size_filter(paintera_path, paintera_key,
                                         boundary_path, boundary_key,
                                         size_threshold, tmp_folder,
                                         target, max_jobs, n_threads,
                                         backup_assignments=True):
    assert False, "TODO"
    assignment_key = 'fragment-segment-assignment'
    ass_tmp_key = 'assignments/initial'
    data_key = 'data/s0'
    g = z5py.File(paintera_path)[paintera_key]
    assert assignment_key in g, "Can't find paintera assignments"
    assert data_key in g, "Can't find paintera data"
    config_dir = os.path.join(tmp_folder, 'configs')

    node_ids, exp_path, chunks = prepare_postprocesing(paintera_path, paintera_key,
                                                       boundary_path, boundary_key,
                                                       data_key, assignment_key,
                                                       ass_tmp_key, backup_assignments,
                                                       tmp_folder, target, max_jobs, n_threads)

    ass_cc_key = 'assignments/connected_components'
    out_cc_key = 'volumes/connected_components'

    task = ConnectedComponentsWorkflow
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             problem_path=exp_path, graph_key='s0/graph',
             path=paintera_path, fragments_key=data_key,
             assignment_path=exp_path,
             assignment_key=ass_tmp_key,
             output_path=exp_path,
             assignment_out_key=ass_cc_key,
             output_key=out_cc_key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Connected components failed"

    # make link to fragments
    frag_tmp_key = 'volumes/fragments'
    trgt = os.path.join(paintera_path, data_key)
    dest = os.path.join(exp_path, frag_tmp_key)
    if not os.path.exists(dest):
        os.symlink(trgt, dest)

    ass_filtered_key = 'assignments/size_filtered'
    task = SizeFilterAndGraphWatershedWorkflow
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             problem_path=exp_path,
             graph_key='s0/graph', features_key='features',
             path=exp_path, segmentation_key=out_cc_key,
             fragments_key=frag_tmp_key,
             assignment_key=ass_cc_key,
             size_threshold=size_threshold,
             relabel=True, output_path=exp_path,
             assignment_out_key=ass_filtered_key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Size filter failed"

    # load the new assignments, bring to paintera format and save
    f = z5py.File(exp_path)
    ds_ass = f[ass_filtered_key]
    ds_ass.n_threads = 8
    new_assignments = ds_ass[:]

    new_assignments = new_assignments[node_ids]
    new_assignments[new_assignments != 0] += node_ids.max()
    new_assignments = np.concatenate([node_ids[:, None], new_assignments[:, None]], axis=1)

    assignment_saver(paintera_path, assignment_key,
                     n_threads, new_assignments, chunks)


def size_filter(paintera_path, paintera_key,
                boundary_path, boundary_key,
                size_threshold, tmp_folder,
                target, max_jobs, n_threads,
                backup_assignments=True):
    assignment_key = 'fragment-segment-assignment'
    data_key = 'data/s0'
    g = z5py.File(paintera_path)[paintera_key]
    assert assignment_key in g, "Can't find paintera assignments"
    assert data_key in g, "Can't find paintera data"

    exp_path = os.path.join(tmp_folder, 'data.n5')
    config_dir = os.path.join(tmp_folder, 'configs')
    seg_key = 'volumes/segmentation'

    # 1.) serialize the current paintera segemntation
    # to get the full volume so we can compute segment sizes
    tmp_serialize = os.path.join(tmp_folder, 'tmp_serialize')
    serialize_from_commit(paintera_path, paintera_key,
                          exp_path, seg_key,
                          tmp_serialize, max_jobs, target,
                          relabel_output=True)

    # 2.) compute graph and weigthts
    compute_graph_and_weights(boundary_path, boundary_key,
                              paintera_path, os.path.join(paintera_key, data_key),
                              exp_path, tmp_folder, target, max_jobs)

    # 3.) save the relabeled assignments in a format that can be ingested by
    # the graph watershed workflow
    relabeled_assignment_path = os.path.join(tmp_serialize, 'assignments.n5')
    relabeled_assignment_key = 'assignments'
    with z5py.File(relabeled_assignment_path, 'r') as f:
        ds_relabeled = f[relabeled_assignment_key]
        ds_relabeled.n_threads = n_threads
        relabeled_assignments = ds_relabeled[:, 1]

    f = z5py.File(exp_path)
    new_assignment_key = 'assignments/relabeled_assignments'
    chunks1d = (min(len(relabeled_assignments), 1000000),)
    ds_out = f.require_dataset(new_assignment_key, shape=relabeled_assignments.shape, chunks=chunks1d,
                               compression='gzip', dtype='uint64')
    ds_out[:] = relabeled_assignments

    # 4.) run size filter work-flow
    ass_filtered_key = 'assignments/size_filtered'
    task = SizeFilterAndGraphWatershedWorkflow
    t = task(tmp_folder=tmp_folder, config_dir=config_dir,
             max_jobs=max_jobs, target=target,
             problem_path=exp_path,
             graph_key='s0/graph', features_key='features',
             path=exp_path, segmentation_key=seg_key,
             assignment_key=new_assignment_key,
             size_threshold=size_threshold,
             relabel=False, output_path=exp_path,
             assignment_out_key=ass_filtered_key)
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Size filter failed"

    # 5.) backup the chunks if specified
    ff = z5py.File(paintera_path)
    ds_ass = ff[paintera_path][assignment_key]
    chunks = ds_ass.chunks
    if backup_assignments:
        bkp_key = os.path.join(paintera_key, 'assignments-bkp')
        ds_ass.n_threads = n_threads
        assignments = ds_ass[:]
        print("Making back-up @", paintera_path, ":", bkp_key)
        assignment_saver(paintera_path, bkp_key, n_threads,
                         assignments, chunks)

    # 6.) load the new assignments, bring to paintera format and save
    f = z5py.File(exp_path)
    ds_ass = f[ass_filtered_key]
    ds_ass.n_threads = n_threads
    new_assignments = ds_ass[:]

    new_assignments[1:] += len(new_assignments)
    new_assignments = np.concatenate([np.arange(len(new_assignments), dtype='uint64')[:, None],
                                      new_assignments[:, None]], axis=1)
    assignment_saver(paintera_path, os.path.join(paintera_key, assignment_key),
                     n_threads, new_assignments, chunks)
