import os
import json
import numpy as np

import luigi
import z5py
from cluster_tools.write import WriteLocal, WriteSlurm


# TODO proprly implement this
def parse_actions(project_dir, dataset_name=None):
    attrs = os.path.join(project_dir, 'attributes.json')
    with open(attrs, 'r') as f:
        attrs = json.load(f)

    datasets = attrs['paintera']['sourceInfo']['sources']
    label_datasets = [ds['state'] for ds in datasets if ds['type'].split('.')[-1] == 'LabelSourceState']
    assert len(label_datasets) > 0

    if dataset_name is None:
        assert len(label_datasets) == 1
        dataset = label_datasets[0]
    else:
        dataset_names = [ds['name'] for ds in label_datasets]
        assert dataset_name in dataset_names
        dataset = [ds for ds in label_datasets if ds['name'] == dataset_name][0]

    # load n5 label dataset
    source = dataset['source']['source']['meta']
    # TODO support hdf5 to
    path, key = source['n5'], source['dataset']
    print("N5-file corresponding to dataset:", path, ":", key)

    # load assignments
    assignments = dataset['assignment']
    # TODO handle other types
    ass_type = assignments['type'].split('.')[-1]
    assert ass_type == 'FragmentSegmentAssignmentOnlyLocal'
    assignments = assignments['data']
    actions = assignments['actions']

    # check if we have an inital lut
    initial_lut = assignments['initialLut']['data']['N5']['data']
    lut_path, lut_key = initial_lut['n5'], initial_lut['dataset']
    with z5py.File(lut_path, 'r') as f:
        have_lut = lut_key in f
    # TODO handle initial lut
    assert have_lut, "Don't support LUT"

    # load actions
    # print(actions[:10])

    # load canvas
    # print(list(dataset.keys()))
    print(dataset['source'])
    return actions[:]


def serialize_from_actions():
    pass


def serialize_assignments(g, ass_key, save_path, save_key):
    # load the assignments
    ass = g[ass_key][:].T

    # get the max overall id
    n_ids = int(ass.max()) + 1
    # cast to dense assignments
    dense_assignments = np.zeros((n_ids + 1, 2), dtype='uint64')
    dense_assignments[:-1, 0] = np.arange(n_ids)
    dense_assignments[:-1, 1] = np.arange(n_ids)
    dense_assignments[:-1, 1][ass[:, 0]] = ass[:, 1]
    # add assignment for the paintera ignore label
    dense_assignments[-1, 0] = 18446744073709551615
    dense_assignments[-1, 1] = 0

    # save the dense assignments in temp folder
    chunks = (min(n_ids, 1000000), 2)
    f_ass = z5py.File(save_path)
    ds = f_ass.require_dataset(save_key, shape=dense_assignments.shape, chunks=chunks,
                               compression='gzip', dtype='uint64')
    ds[:] = dense_assignments


def serialize_merged_segmentation(path, key, out_path, out_key, ass_path, ass_key,
                                  tmp_folder, max_jobs, target):
    task = WriteLocal if target == 'local' else WriteSlurm
    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)
    global_config = task.default_global_config()

    with z5py.File(path) as f:
        block_shape = f[key].chunks

    # TODO allow specifying the shebang
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    global_config['shebang'] = shebang
    global_config.update({'shebang': shebang, 'block_shape': block_shape})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    config = task.default_task_config()
    config.update({'chunks': block_shape})
    with open(os.path.join(config_folder, 'write.config'), 'w') as f:
        json.dump(config, f)

    t = task(tmp_folder=tmp_folder, config_dir=config_folder, max_jobs=max_jobs,
             input_path=path, input_key=key,
             output_path=out_path, output_key=out_key,
             assignment_path=ass_path, assignment_key=ass_key,
             identifier='merge-paintera-seg')
    ret = luigi.build([t], local_scheduler=True)
    assert ret, "Writing mereged segmentation failed"


def serialize_from_commit(path, key, out_path, out_key,
                          tmp_folder, max_jobs, target):
    f = z5py.File(path, 'r')
    g = f[key]

    # make sure this is a paintera group
    seg_key = 'data'
    ass_key = 'fragment-segment-assignment'
    assert seg_key in g
    assert ass_key in g

    os.makedirs(tmp_folder, exist_ok=True)
    save_path = os.path.join(tmp_folder, 'assignments.n5')
    save_key = 'assignments'
    print("Serializing assignments ...")
    serialize_assignments(g, ass_key, save_path, save_key)

    full_seg_key = os.path.join(key, seg_key, 's0')
    print("Serializing new segmentation ...")
    serialize_merged_segmentation(path, full_seg_key,
                                  out_path, out_key,
                                  save_path, save_key,
                                  tmp_folder, max_jobs, target)
