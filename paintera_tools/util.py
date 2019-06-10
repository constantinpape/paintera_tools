import os
import json
import numpy as np
import luigi
import z5py

from cluster_tools.relabel import UniqueWorkflow
from cluster_tools.workflows import ProblemWorkflow

from . import default_config


def save_assignments(assignments, save_path, save_key):
    # save the dense assignments in temp folder
    chunks = (min(len(assignments), 1000000), 2)
    f_ass = z5py.File(save_path)
    ds = f_ass.require_dataset(save_key, shape=assignments.shape, chunks=chunks,
                               compression='gzip', dtype='uint64')
    ds[:] = assignments


def make_dense_assignments(fragment_ids, assignments):

    assignment_dict = dict(zip(assignments[:, 0], assignments[:, 1]))
    dense_assignments = {frag_id: assignment_dict.get(frag_id, frag_id) for frag_id in fragment_ids}

    # set the paintera ignore label assignment to 0
    paintera_ignore_label = 18446744073709551615
    if paintera_ignore_label in dense_assignments:
        dense_assignments[paintera_ignore_label] = 0

    frag_ids = np.array(list(dense_assignments.keys()), dtype='uint64')
    seg_ids = np.array(list(dense_assignments.values()), dtype='uint64')
    dense_assignments = np.concatenate([frag_ids[:, None], seg_ids[:, None]], axis=1)
    assert dense_assignments.shape[1] == 2
    return dense_assignments


def find_uniques(path, seg_in_key, out_path, out_key,
                 tmp_folder, config_folder, max_jobs, target):
    task = UniqueWorkflow
    t = task(tmp_folder=tmp_folder, config_dir=config_folder,
             max_jobs=max_jobs, target=target,
             input_path=path, input_key=seg_in_key,
             output_path=out_path, output_key=out_key)
    ret = luigi.build([t], local_scheduler=True)
    if not ret:
        raise RuntimeError("Find uniques failed")


def write_global_config(config_folder, block_shape=None):
    os.makedirs(config_folder, exist_ok=True)
    global_config = UniqueWorkflow.get_config()['global']

    block_shape = default_config.get_default_block_shape() if block_shape is None else block_shape
    global_config.update({'shebang': default_config.get_default_shebang(),
                          'group': default_config.get_default_group(),
                          'block_shape': block_shape})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)


def compute_graph_and_weights(aff_path, aff_key,
                              seg_path, seg_key, out_path,
                              tmp_folder, target, max_jobs,
                              offsets=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]):
    config_folder = os.path.join(tmp_folder, 'configs')
    chunks = z5py.File(seg_path, 'r')[seg_key].chunks
    write_global_config(config_folder, chunks)

    configs = ProblemWorkflow.get_config()
    conf = configs['block_edge_features']
    conf.update({'mem_limit': 4})
    if offsets is not None:
        conf.update({'offsets': offsets})
    with open(os.path.join(config_folder, 'block_edge_features.config'), 'w') as f:
        json.dump(conf, f)

    conf_names = ['merge_sub_graphs', 'map_edge_ids', 'merge_edge_features']
    # TODO make this configurable
    n_threads = 12
    max_ram = 64
    for name in conf_names:
        conf = configs[name]
        conf.update({'threads_per_job': n_threads, 'mem_limit': max_ram})
        with open(os.path.join(config_folder, '%s.config' % name), 'w') as f:
            json.dump(conf, f)

    task = ProblemWorkflow(tmp_folder=tmp_folder, config_dir=config_folder,
                           target=target, max_jobs=max_jobs,
                           input_path=aff_path, input_key=aff_key,
                           ws_path=seg_path, ws_key=seg_key,
                           problem_path=out_path)
    ret = luigi.build([task], local_scheduler=True)
    assert ret, "Problem extraction failed"
