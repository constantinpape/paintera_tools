import os
from z5py.util import copy_group, copy_dataset


def make_initial_dataset(out_path):
    path = '/g/kreshuk/data/schwab/parapodia_rachel/data.n5'
    in_key = 'volumes/paintera/lmc_nuclei_v1_fragments'
    out_key = 'volumes/segmentation'
    # copy the paintera dataset
    # FIXME this memory errs
    print("Start copying...")
    copy_group(path, out_path, in_key, out_key, 8)
    # make link to raw data
    os.symlink(os.path.join(path, 'volumes/raw'),
               os.path.join(out_path, 'volumes/raw'))


def check(out_path):
    path = '/g/kreshuk/data/schwab/parapodia_rachel/data.n5'
    in_key = 'volumes/paintera/lmc_nuclei_v1_fragments/label-to-block-mapping/s0'
    out_key = 'volumes/segmentation/labs'
    copy_dataset(path, out_path, in_key, out_key, 8)


if __name__ == '__main__':
    path = '/g/schwab/templin/proofreading/v1_rachel.n5'
    make_initial_dataset(path)
