import os
import json
import numpy as np
from concurrent import futures
from threading import Lock
from copy import deepcopy

import z5py
import vigra
import nifty
import nifty.distributed as ndist
import nifty.tools as nt
from z5py.util import copy_dataset

from ..util import compute_graph_and_weights, make_dense_assignments, find_uniques


def load_paintera_assignments(paintera_path, data_key, assignment_key,
                              exp_path, tmp_folder, max_jobs, target):
    tmp_key = 'uniques'
    # NOTE the config is already written by conpute_graph_and_weights
    config_folder = os.path.join(tmp_folder, 'configs')
    find_uniques(paintera_path, data_key, exp_path, tmp_key,
                 tmp_folder, config_folder, max_jobs, target)
    # load the unique ids
    f = z5py.File(exp_path)
    fragment_ids = f[tmp_key][:]

    # load the assignments
    f = z5py.File(paintera_path)
    assignments = f[assignment_key][:].T
    return make_dense_assignments(fragment_ids, assignments)


def prepare_splitter(paintera_path, paintera_key, boundary_path, boundary_key,
                     exp_path, tmp_folder, target, max_jobs, backup_assignments=True):
    assignment_key = 'fragment-segment-assignments'
    data_key = 'data/s0'
    g = z5py.File(paintera_path)[paintera_key]
    assert assignment_key in g, "Can't find paintera assignments"
    assert data_key in g, "Can't find paintera data"

    # make backup of assignments
    if backup_assignments:
        bkp_key = os.path.join(paintera_path, paintera_key, 'assignments-bkp')
        print("Making back-up @", bkp_key)
        copy_dataset(paintera_path, os.path.join(paintera_key, assignment_key),
                     paintera_path, bkp_key, 4)

    # make the problem
    compute_graph_and_weights(boundary_path, boundary_key,
                              paintera_path, data_key, exp_path,
                              tmp_folder, target, max_jobs)

    # load the assignments
    assignments = load_paintera_assignments(paintera_path, data_key, assignment_key,
                                            exp_path, tmp_folder, max_jobs, target)
    return assignments


def print_help():
    print("Interactive paintera splitting:")
    print("[s]: enter split-mode")
    print("[q]: end interactive splitting")
    print("[h]: display help message")
    print("Split-mode:")
    print("[n]: start new seed")
    print("[g <id>] go back to seed number <id>")
    print("[s]: run watershed segmentation and save")
    print("[u]: undo last seeds and segmentation")
    print("[q]: return to interactive mode")
    print("[p]: save current seeds to json")


def isint(x):
    try:
        int(x)
        return True
    except Exception:
        return False


def to_seeds(seeds_to_fragments):
    seed_list = [seeds_to_fragments[seed_id]
                 for seed_id in sorted(seeds_to_fragments.keys())]
    if not all(seed_list):
        return None
    return seed_list


def save_seeds(save_path, seeds_to_fragments):
    print("saving current seeds to", save_path)
    with open(save_path, 'w') as f:
        json.dump(seeds_to_fragments, f)


def split_mode(splitter, assignments, ds_assignments):
    print("Entering split-mode")
    segment_id = input("enter the segment id of object to be split")
    try:
        segment_id = int(segment_id)
    except ValueError:
        print("Invalid input for segment id, leaving split-mode")
        return
    print("Splitting segment", segment_id)

    # seed storage
    seeds_to_fragments = {1: []}
    current_seed_id = 1
    max_seed_id = 1

    # last step backup for undo functionality
    previous_assignments = assignments.copy()
    previous_seeds = deepcopy(seeds_to_fragments)

    save_path = 'split_state_object_%i.json' % segment_id
    while True:
        print("Split-mode: seeds and fragments", seeds_to_fragments)
        print("Split-mode: current seed", current_seed_id)
        x = input("Split-mode: waiting for input")
        if isint(x):
            seeds_to_fragments[current_seed_id].append(int(x))

        elif x == 'n':
            max_seed_id += 1
            current_seed_id = max_seed_id
            seeds_to_fragments[current_seed_id] = []

        elif x == 's':
            previous_assignments = assignments.copy()
            previous_seeds = deepcopy(seeds_to_fragments)
            seeds = to_seeds(seeds_to_fragments)
            if seeds is None:
                print("Split-mode: at least one seed is empty, aborting split action")
                continue
            assignments = splitter.split_segment(segment_id, seeds, assignments)
            ds_assignments[:] = assignments
            print("Split-mode: have split segment and written new assignments, please reload paintera.")

        elif x == 'u':
            seeds_to_fragments = previous_seeds
            assignments = previous_assignments

        elif x == 'q':
            x = input("Do you really want to quit split-mode? y / [n]")
            if x.lower() == 'y':
                break

        elif x.startswith('g'):
            try:
                seed_id = int(x[1:])
            except ValueError:
                print("Split-mode: Invaild input", x, "enter [h] for help")
                continue
            if seed_id < max_seed_id:
                print("Split-mode: cannot select seed", seed_id)
                continue
            current_seed_id = seed_id

        elif x == 'h':
            print_help()

        elif x == 'p':
            save_seeds(save_path, seeds_to_fragments)

        else:
            print("Split-mode: Invaild input", x, "enter [h] for help")


# TODO delay keyboard interrupt and save when it's called
def interactive_step(splitter, assignments, ds_assignments):
    x = input("Interactive splitting: waiting for input")
    if x == 's':
        split_mode(splitter)
        return True
    elif x == 'q':
        x = input("Do you really want to quit interactive splitting? y / [n]")
        if x.lower() == 'y':
            return False
        else:
            return True
    elif x == 'h':
        print_help()
        return True
    else:
        print("Invaild input", x, "enter [h] for help")
        return True


def interactive_splitter(paintera_path, paintera_key, boundary_path, boundary_key,
                         tmp_folder, target, max_jobs, n_threads):
    """ Interactive splitting of merged paintera objects.

    Arguments:
        paintera_path: path to the paintera project (all changes need to be commited!)
        paintera_key: path in file
        boundary_path: path to n5 file with boundary or affinity maps
        boundary_key: path in file
    """

    print("Start preprocessing for interactive splitter.")
    exp_path = os.path.join(tmp_folder, 'data.n5')
    assignments = prepare_splitter(paintera_path, paintera_key, boundary_path, boundary_key,
                                   exp_path, tmp_folder, target, max_jobs)

    if target != 'local':
        print("Preprocessing on cluster done")
        print("Restart with target set to 'local' in order to start the interactive mode")
        return

    # make splitter
    splitter = Splitter(exp_path, 's0/graph', exp_path, 'features',
                        assignments, n_threads)

    assignment_key = os.path.join(paintera_key, 'fragment-segment-assignments')
    ds_assignments = z5py.File(paintera_path)[assignment_key]
    ds_assignments.n_threads = n_threads

    print("Start interactive splitting session")
    print_help()

    # start interactive splitting session
    while True:
        if not interactive_step(splitter):
            break


def batch_splitter(paintera_path, paintera_key, boundary_path, boundary_key,
                   segment_ids, all_seed_fragments,
                   tmp_folder, target, max_jobs, n_threads):
    exp_path = os.path.join(tmp_folder, 'data.n5')
    assignments = prepare_splitter(paintera_path, paintera_key, boundary_path, boundary_key,
                                   exp_path, tmp_folder, target, max_jobs)
    # TODO implement this on the cluster
    if target != 'local':
        raise NotImplementedError("Batch splitting is only implemented locally.")

    # make splitter
    splitter = Splitter(exp_path, 's0/graph', exp_path, 'features',
                        assignments, n_threads)
    assignments = splitter.split_multiple_segments(segment_ids, all_seed_fragments,
                                                   assignments, n_threads)

    assignment_key = os.path.join(paintera_key, 'fragment-segment-assignments')
    ds_assignments = z5py.File(paintera_path)[assignment_key]
    ds_assignments.n_threads = n_threads
    ds_assignments[:] = assignments


# - split segment from seeds
# - split multiple segments in paralellel for batch mode
# TODO implement different splitting methods (LMC)
class Splitter:
    def __init__(self, graph_path, graph_key,
                 weights_path, weights_key,
                 n_threads):
        self.n_threads = n_threads

        # load graph and weights
        self.graph = ndist.Graph(os.path.join(graph_path, graph_key), n_threads)
        self.uv_ids = self.graph.uvIds()

        weight_ds = z5py.File(weights_path)[weights_key]
        weight_ds.n_threads = self.n_threads
        self.weights = weight_ds[:, 0].squeeze() if weight_ds.ndim == 2 else weight_ds[:]
        assert len(self.weights) == self.graph.numberOfEdges

    #
    # split segment functionality
    #

    # impl does not change the state and can be run in parallel
    def _split_segment_impl(self, fragment_ids, seed_fragments):
        sub_edges, _ = self.graph.extractSubgraphFromNodes(fragment_ids,
                                                           allowInvalidNodes=True)
        sub_uvs = self.uv_ids[sub_edges]
        sub_weights = self.weights[sub_edges]
        assert len(sub_edges) == len(sub_weights)

        # relabel the local fragment ids
        nodes, max_id, mapping = vigra.analysis.relabelConsecutive(fragment_ids,
                                                                   start_label=0,
                                                                   keep_zeros=False)
        sub_uvs = nt.takeDict(mapping, sub_uvs)
        n_sub_nodes = max_id + 1
        # build watershed problem and run watershed
        sub_graph = nifty.graph.undirectedGraph(n_sub_nodes)
        sub_graph.insertEdges(sub_uvs)

        # make seeds
        sub_seeds = np.zeros(n_sub_nodes, dtype='uint64')
        seed_id = 1
        # TODO vectorize
        for seed_group in seed_fragments:
            for seed_fragment in seed_group:
                mapped_id = mapping[seed_fragment]
                sub_seeds[mapped_id] = seed_id
            seed_id += 1

        # TODO support other splitting options, e.g. LMC
        # run graph watershed
        sub_assignment = nifty.graph.edgeWeightedWatershedsSegmentation(sub_graph,
                                                                        sub_seeds,
                                                                        sub_weights)
        assert len(sub_assignment) == n_sub_nodes == len(fragment_ids)
        return sub_assignment

    def split_segment(self, segment_id, seed_fragments, assignments):
        assert isinstance(segment_id, int)
        assert isinstance(seed_fragments, list)
        # TODO not really sure that this is the correct check
        assert len(assignments) == self.graph.numberOfNodes, "%i, %i" % (len(assignments),
                                                                         self.graph.numberOfNodes)
        max_id = int(assignments.max())

        # get fragments and segment mask
        segment_mask = assignments[1] == segment_id
        fragment_ids = assignments[0][segment_mask]
        if not fragment_ids.size:
            return None

        # split the segment
        split_assignments = self._split_segment_impl(fragment_ids, seed_fragments)

        # offset the split_assignments
        split_assignments[split_assignments != 0] += max_id
        assignments[segment_mask] = split_assignments
        return assignments

    def split_multiple_segments(self, segment_ids, all_seed_fragments, assignments, n_threads):
        assert isinstance(segment_ids, int)
        assert isinstance(all_seed_fragments, list)
        assert len(segment_ids) == len(all_seed_fragments)
        # TODO not really sure that this is the correct check
        assert len(assignments) == self.graph.numberOfNodes, "%i, %i" % (len(assignments),
                                                                         self.graph.numberOfNodes)
        lock = Lock()

        def _split(segment_id, seed_fragments):
            segment_mask = assignments[1] == segment_id
            fragment_ids = assignments[0][segment_mask]
            split_assignments = self._split_segment_impl(fragment_ids,
                                                         seed_fragments)
            with lock:
                max_id = int(assignments.max())
                split_assignments[split_assignments != 0] += max_id
                assignments[segment_mask] = split_assignments

        with futures.ThreadPoolExecutor(n_threads) as tp:
            tasks = [tp.submit(_split, segment_id, seed_fragments)
                     for segment_id, seed_fragments in zip(segment_ids,
                                                           all_seed_fragments)]
            [t.result() for t in tasks]
        return assignments
