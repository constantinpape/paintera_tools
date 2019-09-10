import nifty.tools as nt
import numpy as np
import z5py
from elf.label_multiset import deserialize_multiset
from tqdm import trange


def check_serialization(mset1, mset2):

    if len(mset1) != len(mset2):
        print("Serialization sizes disagree:", len(mset1), len(mset2))
        return False

    if not np.array_equal(mset1, mset2):
        disagree = (mset1 != mset2)
        print("Serializations disagree for entries", disagree.sum(), "/", disagree.size)
        return False

    print("Check serialization passed")
    return True


def check_multiset_members(mset1, mset2):
    assert mset1.shape == mset2.shape
    if mset1.n_elements != mset2.n_elements:
        print("N-elements disagree:", mset1.n_elements, mset2.n_elements)
        return False

    amax1, amax2 = mset1.argmax, mset2.argmax
    if not np.array_equal(amax1, amax2):
        disagree = (amax1 != amax2)
        print("Argmax disagree for entries", disagree.sum(), "/", disagree.size)
        return False

    off1, off2 = mset1.offsets, mset2.offsets
    if not np.array_equal(off1, off2):
        disagree = (off1 != off2)
        print("Offsets disagree for entries", disagree.sum(), "/", disagree.size)
        return False

    id1, id2 = mset1.ids, mset2.ids
    if not np.array_equal(id1, id2):
        disagree = (id1 != id2)
        print("Ids disagree for entries", disagree.sum(), "/", disagree.size)
        return False

    count1, count2 = mset1.counts, mset2.counts
    if not np.array_equal(count1, count2):
        disagree = (count1 != count2)
        print("Counts disagree for entries", disagree.sum(), "/", disagree.size)
        return False

    print("Check members passed")
    return True


def check_pixels(mset1, mset2, seg, scale):
    roi_end = mset1.shape
    # roi_end = [10, 10, 10]
    blocking = nt.blocking([0, 0, 0], roi_end, [1, 1, 1])
    for block_id in trange(blocking.numberOfBlocks):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        i1, c1 = mset1[bb]
        i2, c2 = mset2[bb]

        if not np.array_equal(i1, i2) or not np.array_equal(c1, c2):
            print("Entries disagree for coords", bb)
            print("Ids")
            print("Res:", i1)
            print("Exp:", i2)
            print("Counts")
            print("Res:", c1)
            print("Exp:", c2)

            print("From segmentation")
            effective_bb = tuple(slice(b.start * sc, b.stop * sc) for b, sc in zip(bb, scale))
            print(effective_bb)
            sub_seg = seg[effective_bb]
            sids, scounts = np.unique(sub_seg, return_counts=True)
            print("Ids")
            print(sids)
            print("Counts")
            print(scounts)

            return False

    print("Check pixels passed")
    return True


def check_chunk(blocking, chunk_id, ds_mset1, ds_mset2, ds_seg, scale):
    block = blocking.getBlock(chunk_id)
    chunk = tuple(beg // ch for beg, ch in zip(block.begin, blocking.blockShape))
    mset1 = ds_mset1.read_chunk(chunk)
    mset2 = ds_mset2.read_chunk(chunk)

    if(check_serialization(mset1, mset2)):
        print("Multisets agree")
        return

    mset1 = deserialize_multiset(mset1, block.shape)
    mset2 = deserialize_multiset(mset2, block.shape)
    if(check_multiset_members(mset1, mset2)):
        print("Multisets agree")
        return

    ds_seg.n_threads = 8
    seg = ds_seg[:]
    if(check_pixels(mset1, mset2, seg, scale)):
        print("Multisets agree")
    else:
        print("Multisets disagree")


def get_scale(level):
    scales = [[1, 1, 1],
              [1, 2, 2],
              [1, 4, 4],
              [2, 8, 8]]
    return scales[level]


def check_multiset(level):
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    seg_key = 'volumes/segmentation/multicut'
    mset_key = 'paintera/data/s%i' % level

    f = z5py.File(path)
    ds_seg = f[seg_key]
    ds_mset = f[mset_key]

    path1 = '/home/pape/Work/data/cremi/example/sampleA_paintera.n5'
    mset_key1 = 'volumes/segmentation/multicut/data/s%i' % level
    f1 = z5py.File(path1)
    ds_mset1 = f1[mset_key1]
    assert ds_mset.shape == ds_mset1.shape
    assert ds_mset.chunks == ds_mset1.chunks, "%s, %s" % (str(ds_mset.chunks),
                                                          str(ds_mset1.chunks))
    shape, chunks = ds_mset.shape, ds_mset.chunks

    blocking = nt.blocking([0, 0, 0], shape, chunks)
    scale = get_scale(level)
    check_chunk(blocking, 0, ds_mset, ds_mset1, ds_seg, scale)


if __name__ == '__main__':
    print("Checking mult-sets for chunk 0 of scales:")
    for scale in range(4):
        print("Check scale", scale)
        check_multiset(scale)
