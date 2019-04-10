import os
import json
import numpy as np

import z5py
import luigi
import vigra
import nifty
import nifty.distributed as ndist
import nifty.tools as nt
from cluster_tools.workflows import ProblemWorkflow


def compute_graph_and_weights(aff_path, aff_key, seg_path, seg_key, out_path,
                              tmp_folder, target, max_jobs,
                              offsets=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]):
    os.makedirs(tmp_folder, exist_ok=True)
    config_folder = os.path.join(tmp_folder, 'configs')
    os.makedirs(config_folder, exist_ok=True)

    configs = ProblemWorkflow.get_config()
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    chunks = z5py.File(seg_path, 'r')[seg_key].chunks
    global_config = configs['global']
    global_config['shebang'] = shebang
    global_config.update({'shebang': shebang, 'block_shape': chunks})
    with open(os.path.join(config_folder, 'global.config'), 'w') as f:
        json.dump(global_config, f)

    conf = configs['block_edge_features']
    conf.update({'offsets': offsets})
    with open(os.path.join(config_folder, 'block_edge_features.config'), 'w') as f:
        json.dump(conf, f)

    conf_names = ['merge_sub_graphs', 'map_edge_ids', 'merge_edge_features']
    # TODO make this configurable
    n_threads = 12
    max_ram = 32
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


def split_by_watershed(assignment_path, assignment_key,
                       problem_path, graph_key, feature_key,
                       seed_fragments, n_threads=8):
    # load graph and features
    print("Load graph ...")
    graph = ndist.Graph(os.path.join(problem_path, graph_key), n_threads)
    uv_ids = graph.uvIds()

    print("Load weights ...")
    with z5py.File(problem_path, 'r') as f:
        feat_ds = f[feature_key]
        feat_ds.n_threads = n_threads
        weights = feat_ds[:, 0].squeeze()
    assert len(weights) == graph.numberOfEdges

    # load the assignments
    print("Load assignments ...")
    with z5py.File(assignment_path, 'r') as f:
        ds = f[assignment_key]
        ds.n_threads = n_threads
        assignments = ds[:]
    offset = int(assignments.max()) + 1

    # find the segment id of all seed fragments
    flat_seed_fragments = [seed_frag for seed_frags in seed_fragments
                           for seed_frag in seed_frags]
    seed_mask = np.in1d(assignments[0], flat_seed_fragments)
    segment_ids = np.unique(assignments[1][seed_mask])
    print(segment_ids)
    # TODO for now we only support a single segment, but we should enable multiple
    assert len(segment_ids) == 1

    def split_segment(segment_id):
        print("Splitting segment", segment_id)
        # find all fragment ids belonging to this fragment and extract the corresponding sub-graph
        fragment_mask = assignments[1] == segment_id
        fragment_ids = assignments[0][fragment_mask]
        sub_edges, _ = graph.extractSubgraphFromNodes(fragment_ids, allowInvalidNodes=True)
        sub_uvs = uv_ids[sub_edges]
        sub_weights = weights[sub_edges]
        assert len(sub_edges) == len(sub_weights)

        # relableb the local fragment ids
        nodes, max_id, mapping = vigra.analysis.relabelConsecutive(fragment_ids,
                                                                   start_label=0,
                                                                   keep_zeros=False)
        sub_uvs = nt.takeDict(mapping, sub_uvs)
        n_sub_nodes = max_id + 1
        # build watershed problem and run watershed
        sub_graph = nifty.graph.undirectedGraph(n_sub_nodes)
        sub_graph.insertEdges(sub_uvs)

        # get cureent seeds # TODO
        this_seeds = seed_fragments
        sub_seeds = np.zeros(n_sub_nodes, dtype='uint64')
        seed_id = 1
        for seed_frags in this_seeds:
            for seed_frag in seed_frags:
                mapped_id = mapping[seed_frag]
                sub_seeds[mapped_id] = seed_id
            seed_id += 1

        sub_assignment = nifty.graph.edgeWeightedWatershedsSegmentation(sub_graph, sub_seeds, sub_weights)
        mask = sub_assignment != 0
        return fragment_ids[mask], sub_assignment[mask]

    sub_ids, sub_assignments = split_segment(segment_ids[0])
    sub_assignments += offset
    return segment_ids, sub_ids, sub_assignments
