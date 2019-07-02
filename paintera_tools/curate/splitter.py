import os
import json
import numpy as np
from functools import partial

import z5py
import vigra
import nifty
import nifty.distributed as ndist
import nifty.tools as nt

from ..util import compute_graph_and_weights, assignment_saver


def prepare_splitter(paintera_path, paintera_key, boundary_path, boundary_key,
                     exp_path, tmp_folder, target, max_jobs, backup_assignments=True):
    assignment_key = 'fragment-segment-assignment'
    data_key = 'data/s0'
    g = z5py.File(paintera_path)[paintera_key]
    assert assignment_key in g, "Can't find paintera assignments"
    assert data_key in g, "Can't find paintera data"

    # make the problem
    data_key = os.path.join(paintera_key, data_key)
    compute_graph_and_weights(boundary_path, boundary_key,
                              paintera_path, data_key,
                              exp_path, tmp_folder, target, max_jobs)

    # make backup of assignments
    assignments = g[assignment_key][:].T
    if backup_assignments:
        bkp_key = os.path.join(paintera_key, 'assignments-bkp')
        print("Making back-up @", paintera_path, ":", bkp_key)
        chunks = g[assignment_key].chunks
        assignment_saver(paintera_path, bkp_key, 1, assignments, chunks)

    return assignments


def print_help():
    print("Interactive paintera splitting:")
    print("[h]: display help message")
    print("[q]: end interactive splitting")
    print("[<segment-id>]: enter split-mode for <segment-id>")
    print("Split-mode:")
    print("[c]: commit changes and return to root mode")
    print("[d <id>]: discard seeds for given id (discards all seeds if no id is given)")
    print("[g <id>] go back to seed number <id>")
    print("[l]: load saved seeds from json")
    print("[n]: start new seed")
    print("[p]: save current seeds to json")
    print("[q]: return to root mode without commiting")
    print("[s]: run watershed segmentation and save")


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
    with open(save_path, 'w') as f:
        json.dump(seeds_to_fragments, f, indent=4, sort_keys=True)


def load_seeds(save_path):
    try:
        with open(save_path) as f:
            seeds_to_fragments = json.load(f)
        seeds_to_fragments = {int(k): v for k, v in seeds_to_fragments.items()}
        return seeds_to_fragments
    except Exception:
        return None


def split_mode(segment_id, splitter, assignments, save_assignments):
    print("Split-mode - Start splitting for segment", segment_id)

    # find valid fragment ids for this segment id
    valid_fragments = assignments[:, 0][assignments[:, 1] == segment_id]
    print("Split-mode - Segment", segment_id, "is made up of", len(valid_fragments), "fragments")

    # seed storage
    seeds_to_fragments = {1: []}
    current_seed_id = 1
    max_seed_id = 1

    # last step backup for undo functionality
    current_assignments = assignments.copy()

    save_path = 'split_state_object_%i.json' % segment_id
    while True:
        print("Split-mode - Seeds and fragments", seeds_to_fragments)
        print("Split-mode - Current seed", current_seed_id)
        x = input("Split-mode - Input fragment id or action: ")
        if isint(x):
            fragment_id = int(x)
            if fragment_id not in valid_fragments:
                print("Split-mode: fragment", fragment_id, "is not part of the current segment", segment_id)
                continue
            seeds_to_fragments[current_seed_id].append(fragment_id)

        elif x == 'n':
            max_seed_id += 1
            current_seed_id = max_seed_id
            seeds_to_fragments[current_seed_id] = []

        elif x.startswith('g'):
            try:
                seed_id = int(x[1:])
            except ValueError:
                print("Split-mode - Invaild input", x, "enter [h] for help")
                continue
            if seed_id > max_seed_id:
                print("Split-mode - Cannot select seed", seed_id)
                continue
            current_seed_id = seed_id

        elif x == 's':
            # split the segment
            seeds = to_seeds(seeds_to_fragments)
            if seeds is None:
                print("Split-mode - At least one seed is empty, aborting split action")
                continue
            split_assignments = splitter.split_segment(segment_id, seeds, assignments)
            if split_assignments is None:
                print("Split mode - Could not split with current seeds")
                continue
            current_assignments = split_assignments
            # write new assignments
            save_assignments(assignments=current_assignments)
            print("Split-mode - Have split segment and written new assignments, reload paintera to update")

        elif x == 'c':
            x = input("Do you want to commit the assignments and quit split-mode? y / [n] ")
            if x.lower() == 'y':
                return current_assignments

        elif x == 'q':
            x = input("Do you want to quit split-mode without committing? y / [n] ")
            if x.lower() == 'y':
                # Need to reset to initial assignments
                save_assignments(assignments=assignments)
                return assignments

        elif x == 'h':
            print_help()

        elif x == 'p':
            print("Split-mode - Saving current seeds to", save_path)
            save_seeds(save_path, seeds_to_fragments)

        elif x == 'l':
            print("Split-mode - Loading seeds from", save_path)
            loaded = load_seeds(save_path)
            if loaded is None:
                print("Split-mode - Could not load seeds, doing nothing")
                continue
            seeds_to_fragments = loaded
            max_seed_id = max(seeds_to_fragments.keys())
            if current_seed_id > max_seed_id:
                current_seed_id = 1

        elif x.startswith('d'):
            if x == 'd':
                y = input("Do you want to clear all seeds? y / [n] ")
                if y.lower() == 'y':
                    seeds_to_fragments = {1: []}
            elif isint(x[1:]):
                seed_id = int(x[1:])
                if seed_id > max_seed_id:
                    print("Split-mode - Cannot select seed", seed_id)
                    continue
                y = input("Do you want to clear seed %i? y / [n] " % seed_id)
                if y.lower() == 'y':
                    seeds_to_fragments[seed_id] = []
            else:
                print("Split-mode - Invaild input", x, "enter [h] for help")
                continue

        else:
            print("Split-mode - Invaild input", x, "enter [h] for help")


# TODO delay keyboard interrupt and save when it's called
def interactive_step(splitter, assignments, save_assignments):
    x = input("Interactive splitting - Input segment id or action: ")
    if isint(x):
        segment_id = int(x)
        if segment_id not in assignments[:, 1]:
            print("Interactive splitting - Segment id", segment_id, "is not part of assignment table")
            return True
        assignments = split_mode(segment_id, splitter, assignments, save_assignments)
        return True
    elif x == 'q':
        x = input("Do you want to quit interactive splitting? y / [n] ")
        if x.lower() == 'y':
            return False
        else:
            return True
    elif x == 'h':
        print_help()
        return True
    else:
        print("Interarctive splitting - Invaild input", x, "enter [h] for help")
        return True


def interactive_splitter(paintera_path, paintera_key, boundary_path, boundary_key,
                         tmp_folder, target, max_jobs, n_threads, ignore_label=None):
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
    splitter = Splitter(exp_path, 's0/graph', exp_path, 'features', n_threads,
                        ignore_label=ignore_label)

    assignment_key = os.path.join(paintera_key, 'fragment-segment-assignment')
    save_assignments = partial(assignment_saver, path=paintera_path, key=assignment_key,
                               n_threads=n_threads)

    print_help()
    # start interactive splitting session
    while True:
        if not interactive_step(splitter, assignments, save_assignments):
            break


def batch_splitter(paintera_path, paintera_key, boundary_path, boundary_key,
                   segment_ids, all_seed_fragments,
                   tmp_folder, target, max_jobs, n_threads,
                   ignore_label=None, backup_assignments=True):
    exp_path = os.path.join(tmp_folder, 'data.n5')
    assignments = prepare_splitter(paintera_path, paintera_key, boundary_path, boundary_key,
                                   exp_path, tmp_folder, target, max_jobs, backup_assignments=backup_assignments)
    # TODO implement this on the cluster
    if target != 'local':
        raise NotImplementedError("Batch splitting is only implemented locally.")

    # make splitter
    splitter = Splitter(exp_path, 's0/graph', exp_path, 'features', n_threads,
                        ignore_label=ignore_label)
    assignments = splitter.split_multiple_segments(segment_ids, all_seed_fragments,
                                                   assignments, n_threads)

    assignment_key = os.path.join(paintera_key, 'fragment-segment-assignment')
    assignment_saver(path=paintera_path, key=assignment_key, n_threads=n_threads,
                     assignments=assignments)


# - split segment from seeds
# - split multiple segments in paralellel for batch mode
# TODO implement different splitting methods (LMC)
class Splitter:
    def __init__(self, graph_path, graph_key,
                 weights_path, weights_key,
                 n_threads, ignore_label=None):
        self.n_threads = n_threads

        # load graph and weights
        self.graph = ndist.Graph(os.path.join(graph_path, graph_key), n_threads)
        self.uv_ids = self.graph.uvIds()

        weight_ds = z5py.File(weights_path)[weights_key]
        weight_ds.n_threads = self.n_threads
        self.weights = weight_ds[:, 0].squeeze() if weight_ds.ndim == 2 else weight_ds[:]
        assert len(self.weights) == self.graph.numberOfEdges

        # we need to set the ignore label to be max repulsve
        if ignore_label is not None:
            ignore_mask = (self.uv_ids == ignore_label).any(axis=1)
            self.weights[ignore_mask] = 1.

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
                # assert seed_fragment in fragment_ids, str(seed_fragment)
                mapped_id = mapping.get(seed_fragment, None)
                # FIXME I don't really know why this would happen, do assignments go stale ?
                if mapped_id is None:
                    print("Warning: could not find seed-fragment", seed_fragment)
                    continue
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
        segment_mask = assignments[:, 1] == segment_id
        fragment_ids = assignments[:, 0][segment_mask]
        if not fragment_ids.size:
            return None

        # split the segment
        split_assignments = self._split_segment_impl(fragment_ids, seed_fragments)

        # offset the split_assignments
        new_assignments = assignments.copy()
        split_assignments[split_assignments != 0] += max_id
        new_assignments[:, 1][segment_mask] = split_assignments
        return new_assignments

    def split_multiple_segments(self, segment_ids, all_seed_fragments, assignments, n_threads):
        assert isinstance(segment_ids, list)
        assert isinstance(all_seed_fragments, list)
        assert len(segment_ids) == len(all_seed_fragments)

        next_id = int(assignments.max()) + 1
        for segment_id, seed_fragments in zip(segment_ids, all_seed_fragments):
            print("Splitting segment id", segment_id)
            print("into %i new segments" % len(seed_fragments))

            segment_mask = assignments[:, 1] == segment_id
            # do nothing if the segment mask is empty
            if segment_mask.sum() == 0:
                print("Did not find any fragments for segment id", segment_id)
                continue
            fragment_ids = assignments[:, 0][segment_mask]
            split_assignments = self._split_segment_impl(fragment_ids,
                                                         seed_fragments)
            split_assignments[split_assignments != 0] += next_id
            assignments[:, 1][segment_mask] = split_assignments
            next_id = int(assignments.max()) + 1

        return assignments
