import sys
import z5py
from cremi_tools.viewer.volumina import view

sys.path.append('../../')


def make_new_seg():
    from paintera_tools import serialize_from_commit
    path = '/g/arendt/pape/proofreading_fib/data.n5'
    key = 'volumes/segmentation'
    out_key = 'volumes/merged/v1'

    target = 'slurm'
    max_jobs = 64
    serialize_from_commit(path, key, path, out_key, 'tmp_pao',
                          max_jobs, target)


def check_new_seg():
    path = '/g/arendt/pape/proofreading_fib/data.n5'
    raw_key = 'volumes/raw/s1'
    seg_key = 'volumes/merged/v1'

    f = z5py.File(path)
    dsr = f[raw_key]
    dsr.n_threads = 8
    shape = dsr.shape
    halo = [50, 512, 512]

    bb = tuple(slice(sh // 2 - ha, sh // 2 + ha)
               for sh, ha in zip(shape, halo))

    ds = f[seg_key]
    assert ds.shape == shape
    ds.n_threads = 8

    raw = dsr[bb].astype('float32')
    seg = ds[bb]

    view([raw, seg])


if __name__ == '__main__':
    check_new_seg()
