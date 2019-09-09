import nifty.tools as nt
import numpy as np
import z5py
from elf.label_multiset import deserialize_multiset


def check_serialization(mset1, mset2):

    if len(mset1) != len(mset2):
        print("Serialization sizes disagree:", len(mset1), len(mset2))
        return

    if not np.array_equal(mset1, mset2):
        disagree = (mset1 != mset2)
        print("Serializations disagree for entries", disagree.sum(), "/", disagree.size)
        return

    print("Check serialization passed")


def check_multiset_members(mset1, mset2):
    pass


def check_pixels(mset1, mset2, seg=None):
    pass


def check_chunk(blocking, chunk_id, ds_mset1, ds_mset2, ds_seg=None):
    block = blocking.getBlock(chunk_id)
    chunk = tuple(beg // ch for beg, ch in zip(block.begin, blocking.blockShape))
    mset1 = ds_mset1.read_chunk(chunk)
    mset2 = ds_mset2.read_chunk(chunk)

    check_serialization(mset1, mset2)

    mset1 = deserialize_multiset(mset1, block.shape)
    mset2 = deserialize_multiset(mset2, block.shape)
    check_multiset_members(mset1, mset2)

    if ds_seg is None:
        seg = None
    else:
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        seg = ds_seg[bb]
        assert seg.shape == mset1.shape
    check_pixels(mset1, mset2, seg)


def check_multiset(scale):
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    seg_key = 'volumes/segmentation/multicut'
    mset_key = 'paintera/data/s%i' % scale

    f = z5py.File(path)
    ds_seg = f[seg_key]
    ds_mset = f[mset_key]

    path1 = '/home/pape/Work/data/cremi/example/sampleA_paintera.n5'
    mset_key1 = 'volumes/segmentation/multicut/data/s%i' % scale
    f1 = z5py.File(path1)
    ds_mset1 = f1[mset_key1]
    assert ds_mset.shape == ds_mset1.shape
    assert ds_mset.chunks == ds_mset1.chunks, "%s, %s" % (str(ds_mset.chunks),
                                                          str(ds_mset1.chunks))
    shape, chunks = ds_mset.shape, ds_mset.chunks

    blocking = nt.blocking([0, 0, 0], shape, chunks)

    if scale == 0:
        check_chunk(blocking, 0, ds_mset, ds_mset1, ds_seg)
    else:
        check_chunk(blocking, 0, ds_mset, ds_mset1)


if __name__ == '__main__':
    check_multiset(0)
