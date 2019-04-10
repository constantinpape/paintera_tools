import sys
import pickle
import numpy as np
import nifty.graph.rag as nrag
import z5py
import luigi
from cremi_tools.viewer.volumina import view
from cluster_tools.morphology import MorphologyWorkflow

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


def make_morphology():
    path = '/g/arendt/pape/proofreading_fib/data.n5'
    key = 'volumes/merged/v1'

    target = 'slurm'
    max_jobs = 64

    outp = './exp_data.n5'
    outk = 'morpho'

    task = MorphologyWorkflow(tmp_folder='./tmp_pao', config_dir='./tmp_pao/configs',
                              target=target, max_jobs=max_jobs,
                              input_path=path, input_key=key,
                              output_path=outp, output_key=outk)
    luigi.build([task], local_scheduler=True)


def extract_bounding_box(segment_id):
    p = './exp_data.n5'
    k = 'morpho'

    # morphp:
    # 0   - id
    # 1   - size
    # 2:5 - com
    # 5:8 - mincoord
    # 8:11 - max coord

    with z5py.File(p) as f:
        ds = f[k]
        ds.n_threads = 8
        coords = ds[:, 5:11]
    coords = coords[segment_id]
    minc = coords[:3]
    maxc = coords[3:]
    bb = tuple(slice(int(mi), int(ma + 1)) for mi, ma in zip(minc, maxc))
    return bb


def precompute():
    from paintera_tools import compute_graph_and_weights
    aff_path = '/g/kreshuk/data/schwab/parapodia_rachel/data.n5'
    aff_key = 'volumes/affinities/s1'
    seg_path = '/mnt/arendt/pape/proofreading_fib/data.n5'
    seg_key = 'volumes/segmentation/data/s0'

    tmp_folder = './tmp_pao'
    out_path = './exp_data.n5'
    target = 'slurm'
    max_jobs = 64
    compute_graph_and_weights(aff_path, aff_key, seg_path, seg_key, out_path,
                              tmp_folder, target, max_jobs)


def fix_merge():
    from paintera_tools import split_by_watershed

    seg_path = '/mnt/arendt/pape/proofreading_fib/data.n5'
    ass_key = 'volumes/segmentation/fragment-segment-assignment'

    # tmp_folder = './tmp_pao'

    out_path = './exp_data.n5'
    # target = 'slurm'
    # max_jobs = 64
    # compute_graph_and_weights(aff_path, aff_key, seg_path, seg_key, out_path,
    #                           tmp_folder, target, max_jobs)

    out_key = 'volumes/segmentationV2/fragment-segment-assignment'
    f = z5py.File(seg_path)
    ds_in = f[ass_key]
    assignments = ds_in[:]

    ds_out = f.require_dataset(out_key, shape=ds_in.shape, chunks=ds_in.chunks, dtype=ds_in.dtype,
                               compression=ds_in.compression)

    compute = True
    if compute:
        seed_fragments = [[3523531], [3980164]]
        changed_segments, changed_ids, changed_assignments = split_by_watershed(seg_path, ass_key, out_path,
                                                                                's0/graph', 'features', seed_fragments)

        with open('./res.pkl', 'wb') as f:
            res = {'segments': changed_segments,
                   'ids': changed_ids,
                   'assignments': changed_assignments}
            pickle.dump(res, f)

    else:
        with open('./res.pkl', 'rb') as f:
            res = pickle.load(f)
            changed_ids = res['ids']
            changed_assignments = res['assignments']

    print(np.unique(changed_assignments))
    sanity_check = np.in1d(changed_assignments, assignments).sum()
    assert sanity_check == 0, str(sanity_check)
    # return

    changed_mask = np.in1d(assignments[0], changed_ids)
    print("before")
    print(np.unique(assignments[1, changed_mask]))
    assignments[1, changed_mask] = changed_assignments
    print("after")
    print(np.unique(assignments[1, changed_mask]))
    ds_out[:] = assignments
    print("Saved new assignments to", seg_path, out_key)


def check(segment_id, bb):
    path = '/g/arendt/pape/proofreading_fib/data.n5'
    key_raw = 'volumes/raw/s1'
    key_merged = 'volumes/merged/v1'
    key_frag = 'volumes/segmentation/data/s0'

    f = z5py.File(path)

    ds = f[key_frag]
    ds.n_threads = 8
    frag = ds[bb]

    with open('./res.pkl', 'rb') as fres:
        res = pickle.load(fres)

    fragment_ids = res['ids']
    assignments = res['assignments']

    n_nodes = int(frag.max()) + 1
    node_labels = np.zeros(n_nodes, dtype='uint64')
    node_labels[fragment_ids] = assignments

    print("Compute rag ...")
    rag = nrag.gridRag(frag, numberOfLabels=n_nodes,
                       numberOfThreads=8)
    print("done")
    new_seg = nrag.projectScalarNodeDataToPixels(rag, node_labels)

    seed_fragments = [3523531, 3980164]
    mask_seeds = (frag == seed_fragments[0]).astype('uint32')
    mask_seeds[frag == seed_fragments[1]] = 2

    ds = f[key_merged]
    ds.n_threads = 8
    seg = ds[bb]

    mask_seg = (seg == segment_id).astype('uint32')

    ds = f[key_raw]
    ds.n_threads = 8
    raw = ds[bb].astype('float32')

    view([raw, frag, seg, mask_seeds, mask_seg, new_seg],
         ['raw', 'fragments', 'segments', 'seed-mask', 'seg-mask', 'curated'])


if __name__ == '__main__':
    # make_new_seg()
    # check_new_seg()

    # precompute()
    fix_merge()

    # make_morphology()
    # segment_id = 5567399
    # bb = extract_bounding_box(segmnt_id)
    # bb = (slice(2439, 3462, None), slice(392, 1259, None), slice(1097, 1663, None))
    # check(segment_id, bb)
