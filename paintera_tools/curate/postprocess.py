import os
import luigi
import z5py
import numpy as np

from cluster_tools.postprocess import SizeFilterAndGraphWatershedWorkflow, ConnectedComponentsWorkflow
from ..util import compute_graph_and_weights
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


def connected_components_and_size_filter(paintera_path, paintera_key,
                                         boundary_path, boundary_key,
                                         size_threshold, tmp_folder,
                                         target, max_jobs, n_threads,
                                         backup_assignments=True):
    assignment_key = 'fragment-segment-assignment'
    data_key = 'data/s0'
    g = z5py.File(paintera_path)[paintera_key]
    assert assignment_key in g, "Can't find paintera assignments"
    assert data_key in g, "Can't find paintera data"

    assignments = g[assignment_key][:].T
    chunks = g[assignment_key].chunks
    # make backup of assignments
    if backup_assignments:
        bkp_key = os.path.join(paintera_key, 'assignments-bkp')
        print("Making back-up @", paintera_path, ":", bkp_key)
        assignment_saver(paintera_path, bkp_key, n_threads,
                         assignments, chunks)

    exp_path = os.path.join(tmp_folder, 'data.n5')
    config_dir = os.path.join(tmp_folder, 'configs')

    compute_graph_and_weights(boundary_path, boundary_key,
                              paintera_path, data_key,
                              exp_path, tmp_folder, target, max_jobs)

    data_key = os.path.join(paintera_key, data_key)
    ass_tmp_key = 'assignments/initial'
    f = z5py.File(exp_path)

    # load the graph node ids (= watershed / fragment ids)
    ds_nodes = f['s0/graph/nodes']
    ds_nodes.n_threads = n_threads
    node_ids = ds_nodes[:]

    # save ass
    if ass_tmp_key not in f:
        make_graph_assignments(f, node_ids, assignments, ass_tmp_key,
                               n_threads)

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

    # load the new assignments, bring to paintera format and
    # save
    ds_ass = f[ass_filtered_key]
    ds_ass.n_threads = 8
    new_assignments = ds_ass[:]

    new_assignments = new_assignments[node_ids]
    new_assignments[new_assignments != 0] += node_ids.max()
    new_assignments = np.concatenate([node_ids[:, None], new_assignments[:, None]], axis=1)

    assignment_saver(paintera_path, assignment_key,
                     n_threads, new_assignments, chunks)
