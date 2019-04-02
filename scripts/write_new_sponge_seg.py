#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python

# TODO refactor this into actual repo

import os
import json
import luigi
import numpy as np
import z5py

from cluster_tools.write import WriteLocal
from cluster_tools.utils.task_utils import DummyTask


def inflate_and_save_lut(path, seg_key, lut_in, lut_out):
    f = z5py.File(path)

    lut = f[lut_in][:]
    print(lut.shape)

    ds_seg = f[seg_key]
    n_labels = int(ds_seg.attrs['maxId']) + 1
    n_frag_ids = int(lut[0, :].max()) + 1

    n_labels = max(n_frag_ids, n_labels)
    print(n_labels)

    new_lut = np.arange(n_labels)
    new_lut = np.concatenate([new_lut[:, None], new_lut[:, None]], axis=1)
    print(new_lut.shape)

    new_lut[lut[0, :], 1] = lut[1, :]
    chunks = (10000, 1)
    f.create_dataset(lut_out, data=new_lut, compression='gzip', chunks=chunks)


def write_new_seg():
    path = '/g/kreshuk/pape/Work/data/group_data/arendt/sponge/data.n5'
    key_seg = 'volumes/paintera/lmc/data/s0'
    key_ass = 'volumes/paintera/lmc/fragment-segment-assignment'
    key_ass_new = 'volumes/paintera/lmc/fragment-segment-assignment-dense'

    inflate_and_save_lut(path, key_seg, key_ass, key_ass_new)

    key_out = 'volumes/segmentation/painera_merged'

    config_folder = 'configs'
    global_config = WriteLocal.default_global_config()
    global_config.update({'shebang':
                          "#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python",
                          'block_shape': [32, 256, 256]})

    # add_failed_blocks(global_config)
    os.makedirs('configs', exist_ok=True)
    with open('configs/global.config', 'w') as f:
        json.dump(global_config, f)

    tmp_folder = './tmp'
    max_jobs = 8
    task = WriteLocal(tmp_folder=tmp_folder, config_dir=config_folder,
                      max_jobs=max_jobs, dependency=DummyTask(),
                      input_path=path, input_key=key_seg,
                      output_path=path, output_key=key_out,
                      assignment_path=path, assignment_key=key_ass_new,
                      identifier='sponge')
    ret = luigi.build([task], local_scheduler=True)
    assert ret


def make_new_project():
    pass


write_new_seg()
