import json
from paintera_tools import interactive_splitter, batch_splitter, postprocess
from paintera_tools import set_default_shebang


def interactive_splitting():
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    set_default_shebang(shebang)

    path = '/g/kreshuk/data/cremi/example/sampleA.n5'
    key = 'paintera'
    aff_key = 'volumes/affinities'

    tmp_folder = 'tmp_split'

    n_jobs = 4
    interactive_splitter(path, key, path, aff_key,
                         tmp_folder, 'local', n_jobs, n_jobs)


def batch_splitting():
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    set_default_shebang(shebang)

    path = '/g/kreshuk/data/cremi/example/sampleA.n5'
    key = 'paintera'
    aff_key = 'volumes/affinities'

    tmp_folder = 'tmp_split'

    segment_id = 198625
    with open('./split_state_object_%i.json' % segment_id) as f:
        seed_dict = json.load(f)

    segment_ids = [segment_id]
    all_seed_fragments = [[seed_dict[seed_id] for seed_id in seed_dict]]

    n_jobs = 4
    batch_splitter(path, key, path, aff_key,
                   segment_ids, all_seed_fragments,
                   tmp_folder, 'local', n_jobs, n_jobs)


def run_postprocessing():
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    set_default_shebang(shebang)

    path = '/g/kreshuk/data/cremi/example/sampleA.n5'
    key = 'paintera'
    aff_key = 'volumes/affinities'

    tmp_folder = 'tmp_pp'

    target = 'local'
    max_jobs = 4
    n_threads = 4

    postprocess(path, key, path, aff_key,
                tmp_folder, target, max_jobs, n_threads,
                size_threshold=1500, label=True,
                backup_assignments=True)


if __name__ == '__main__':
    # interactive_splitting()
    # batch_splitting()
    run_postprocessing()
