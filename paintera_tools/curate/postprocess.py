import os
import json
import luigi
import z5py
import numpy as np

from cluster_tools.postprocess import SizeFilterAndGraphWatershedWorkflow
from cluster_tools.postprocess import ConnectedComponentsWorkflow
from ..util import compute_graph_and_weights, assignment_saver
from ..serialize import serialize_from_commit


def make_graph_assignments(f, node_ids, assignments, out_key, n_threads):
    # make node labels from assignments
    assignment_dict = dict(zip(assignments[:, 0],
                               assignments[:, 1]))
    # we need dense assignments for this to work
    max_id = int(node_ids.max())
    node_labels = np.array([assignment_dict.get(node_id, node_id)
                            for node_id in range(max_id + 1)],
                           dtype='uint64')

    # save temporary node labels
    new_chunks = (min(100000, len(node_labels)),)
    ds = f.require_dataset(out_key, shape=node_labels.shape,
                           chunks=new_chunks, compression='gzip',
                           dtype='uint64')
    ds.n_threads = n_threads
    ds[:] = node_labels


def postprocess(paintera_path, paintera_key,
                boundary_path, boundary_key,
                tmp_folder, target, max_jobs, n_threads,
                size_threshold=0, label=False,
                backup_assignments=True):

    if not label and size_threshold == 0:
        print("Neither size filtering nor label selected; doing nothing")
        return

    assignment_key = 'fragment-segment-assignment'
    data_key = 'data/s0'
    g = z5py.File(paintera_path)[paintera_key]
    assert assignment_key in g, "Can't find paintera assignments"
    assert data_key in g, "Can't find paintera data"

    exp_path = os.path.join(tmp_folder, 'data.n5')
    config_dir = os.path.join(tmp_folder, 'configs')
    current_seg_key = 'volumes/segmentation'

    # 1.) serialize the current paintera segemntation
    # to get the full volume so we can compute segment sizes
    tmp_serialize = os.path.join(tmp_folder, 'tmp_serialize')
    serialize_from_commit(paintera_path, paintera_key,
                          exp_path, current_seg_key,
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
        relabeled_assignments = ds_relabeled[:]

    # need to make assignments completely dense
    node_ids, relabeled_assignments = relabeled_assignments[:, 0], relabeled_assignments[:, 1]
    n_nodes = int(node_ids.max()) + 1
    new_assignments = np.zeros(n_nodes, dtype='uint64')
    new_assignments[node_ids] = relabeled_assignments

    f = z5py.File(exp_path)
    current_ass_key = 'assignments/relabeled_assignments'
    chunks1d = (min(len(relabeled_assignments), 1000000),)
    ds_out = f.require_dataset(current_ass_key, shape=new_assignments.shape,
                               chunks=chunks1d, compression='gzip', dtype='uint64')
    ds_out[:] = new_assignments

    # 4.) run connected components if selected
    if label:
        task = ConnectedComponentsWorkflow
        cc_key = 'assignments/connected_components_assignments'
        cc_seg_key = 'volumes/connected_components'
        t = task(tmp_folder=tmp_folder, config_dir=config_dir,
                 max_jobs=max_jobs, target=target,
                 problem_path=exp_path, graph_key='s0/graph',
                 path=paintera_path,
                 fragments_key=os.path.join(paintera_key, data_key),
                 assignment_path=exp_path,
                 assignment_key=current_ass_key,
                 output_path=exp_path,
                 assignment_out_key=cc_key,
                 output_key=cc_seg_key)
        ret = luigi.build([t], local_scheduler=True)
        assert ret, "Connected components failed"
        current_ass_key = cc_key
        current_seg_key = cc_seg_key

    # 5.) run size filter work-flow if size threshold
    if size_threshold > 0:
        task = SizeFilterAndGraphWatershedWorkflow
        configs = task.get_config()
        conf = configs['graph_watershed_assignments']
        conf.update({'n_threads': n_threads, 'mem_limit': 256, 'time_limit': 240})
        with open(os.path.join(config_dir, 'graph_watershed_assignments.json'), 'w') as f:
            json.dump(conf, f)

        filtered_key = 'assignments/size_filtered'
        t = task(tmp_folder=tmp_folder, config_dir=config_dir,
                 max_jobs=max_jobs, target=target,
                 problem_path=exp_path,
                 graph_key='s0/graph', features_key='features',
                 path=exp_path, segmentation_key=current_seg_key,
                 assignment_key=current_ass_key,
                 size_threshold=size_threshold,
                 relabel=False, output_path=exp_path,
                 assignment_out_key=filtered_key)
        ret = luigi.build([t], local_scheduler=True)
        assert ret, "Size filter failed"
        current_ass_key = filtered_key

    # 6.) backup the assignments if specified
    ff = z5py.File(paintera_path)
    ds_ass = ff[paintera_key][assignment_key]
    chunks = ds_ass.chunks
    if backup_assignments:
        bkp_key = os.path.join(paintera_key, 'assignments-bkp')
        ds_ass.n_threads = n_threads
        assignments = ds_ass[:].T
        assignment_saver(paintera_path, bkp_key, n_threads,
                         assignments, chunks)

    # 7.) load the new assignments, bring to paintera format and save
    f = z5py.File(exp_path)
    ds_ass = f[current_ass_key]
    ds_ass.n_threads = n_threads
    new_assignments = ds_ass[:]

    new_assignments[1:] += len(new_assignments)
    new_assignments = np.concatenate([np.arange(len(new_assignments), dtype='uint64')[:, None],
                                      new_assignments[:, None]], axis=1)
    assignment_saver(paintera_path, os.path.join(paintera_key, assignment_key),
                     n_threads, new_assignments, chunks)
