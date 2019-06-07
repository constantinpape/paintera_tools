import os
import numpy as np
from concurrent import futures

import z5py
import vigra
import nifty
import nifty.distributed as ndist
import nifty.tools as nt

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


def interactive_splitter(paintera_path, paintera_key, boundary_path, boundary_key,
                         tmp_folder, target, max_jobs, n_threads):
    """ Interactive splitting of merged paintera objects.

    Arguments:
        paintera_path: path to the paintera project (all changes need to be commited!)
        paintera_key: path in file
        boundary_path: path to n5 file with boundary or affinity maps
        boundary_key: path in file
    """

    assignment_key = 'fragment-segment-assignments'
    data_key = 'data/s0'
    g = z5py.File(paintera_path)[paintera_key]
    assert assignment_key in g, "Can't find paintera assignments"
    assert data_key in g, "Can't find paintera data"

    # make the problem
    exp_path = os.path.join(tmp_folder, 'data.n5')
    compute_graph_and_weights(boundary_path, boundary_key,
                              paintera_path, data_key, exp_path,
                              tmp_folder, target, max_jobs)

    # load the assignments
    assignments = load_paintera_assignments()

    if target == 'slurm':
        print("Preprocessing with slurm done")
        print("Restart with target set to 'local' in order to start the interactive mode")
        return

    # make splitter
    splitter = Splitter(exp_path, 's0/graph', exp_path, 'features',
                        assignments, n_threads)

    # TODO
    # start interactive splitting session
    # while True:
    #    pass


def batch_splitter():
    pass


# TODO we at least need the following functionality:
# - split segment from seeds (implemented)
# - split multiple segments in paralellel for batch mode
# - undo last operation by resetting to `previous_assignments` for interactive splitting
# TODO implement different splitting methods (LMC)
class Splitter:
    def __init__(self, graph_path, graph_key, weights_path, weights_key,
                 assignments, n_threads):
        self.n_threads = n_threads

        # load graph and weights
        self.graph = ndist.Graph(os.path.join(graph_path, graph_key), n_threads)
        self.uv_ids = self.graph.uvIds()

        weight_ds = z5py.File(weights_path)[weights_key]
        weight_ds.n_threads = self.n_threads
        self.weights = weight_ds[:, 0].squeeze() if weight_ds.ndim == 2 else weight_ds[:]
        assert len(self.weights) == self.graph.numberOfEdges

        # TODO not really sure that this is the correct check
        # check the initial assignments
        assert len(assignments) == self.graph.numberOfNodes, "%i, %i" % (len(assignments),
                                                                         self.graph.numberOfNodes)

        # initialize the state
        self.current_assignments = assignments
        self.previous_assignments = assignments
        self.max_id = int(assignments.max())

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
        sub_assignment = nifty.graph.edgeWeightedWatershedsSegmentation(sub_graph, sub_seeds, sub_weights)
        assert len(sub_assignment) == n_sub_nodes == len(fragment_ids)
        return sub_assignment

    def split_segment(self, segment_id, seed_fragments):
        assert isinstance(segment_id, int)
        assert isinstance(seed_fragments, list)

        # get fragments and segment mask
        assignments = self.current_assignments
        segment_mask = assignments[1] == segment_id
        fragment_ids = assignments[0][segment_mask]
        if not fragment_ids.size:
            return None

        # split the segment
        split_assignments = self._split_segment_impl(fragment_ids, seed_fragments)

        # update the assignments
        self.previous_assignments = assignments.copy()
        # offset the split_assignments
        split_assignments[split_assignments != 0] += self.max_id
        self.max_id = int(split_assignments.max())
        self.current_assignments[segment_mask] = split_assignments
        return self.current_assignments


# def split_by_watershed(assignment_path, assignment_key,
#                        problem_path, graph_key, feature_key,
#                        seed_fragments, n_threads=8):
#
#     # map the seed fragments to segment ids
#     segment_ids_to_seed_groups = {}
#     for group_id, seeds in enumerate(seed_fragments):
#         seed_mask = np.in1d(assignments[0], seeds)
#         segment_id = np.unique(assignments[1][seed_mask])
#         # each seed group should correspond to a single segment
#         assert len(segment_id) == 1, str(segment_id)
#         segment_id = segment_id[0]
#         if segment_id in segment_ids_to_seed_groups:
#             segment_ids_to_seed_groups[segment_id].append(group_id)
#         else:
#             segment_ids_to_seed_groups[segment_id] = [group_id]
#     segment_ids = list(segment_ids_to_seed_groups.keys())
#
#     def split_segment(segment_id):
#         print("Splitting segment", segment_id)
#         seed_groups = segment_ids_to_seed_groups[segment_id]
#         if len(seed_groups) == 1:
#             print("only have a single seed for segment", segment_id, "doing nothing")
#             return None
#
#         # find all fragment ids belonging to this fragment and extract the corresponding sub-graph
#         fragment_mask = assignments[1] == segment_id
#         fragment_ids = assignments[0][fragment_mask]
#         sub_edges, _ = graph.extractSubgraphFromNodes(fragment_ids, allowInvalidNodes=True)
#         sub_uvs = uv_ids[sub_edges]
#         sub_weights = weights[sub_edges]
#
#     with futures.ThreadPoolExecutor(n_threads) as tp:
#         tasks = [tp.submit(split_segment, seg_id) for seg_id in segment_ids]
#         results = [t.result() for t in tasks]
#     results = [res for res in results if res is not None]
#     ids = np.concatenate([res[0] for res in results], axis=0)
#
#     # make the new assignments
#     new_assignments = []
#     offset = int(assignments.max())
#     # offset by old assignment max id
#     for res in results:
#         ass = res[1]
#         ass += offset
#         new_assignments.append(ass)
#         offset = ass.max()
#     new_assignments = np.concatenate(new_assignments, axis=0)
#     assert len(ids) == len(new_assignments)
#     return ids, new_assignments
