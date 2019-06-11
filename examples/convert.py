from paintera_tools import convert_to_paintera_format
from paintera_tools import set_default_shebang


def convert_cremi(with_assignments):
    path = '/g/kreshuk/data/cremi/example/sampleA.n5'

    # NOTE: the raw data needs to be multiscale, i.e. 'raw_key' needs to be a group with
    # datasets 'raw_key/s0' ... 'raw_key/sN'. It's ok if there's only a single scale, i.e. 'raw_key/s0'
    raw_key = 'raw'

    # we have two options: create a dataset with or without fragment-segment assignments
    if with_assignments:
        # with assignments: we need an assignment path and key
        # and we use the watersheds as input segmentation
        ass_path = path
        ass_key = 'node_labels'
        in_key = 'segmentation/watershed'
    else:
        # without assignments: no assignment path or key
        # and we use the multicut segmentation as input segmentation
        ass_path = ass_key = ''
        in_key = 'segmentation/multicut'

    # output key: we stote the new paintera dataset here
    out_key = 'paintera'

    # shebang to environment with all necessary dependencies
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    set_default_shebang(shebang)

    res = [40, 4, 4]
    tmp_folder = './tmp_convert'
    convert_to_paintera_format(path, raw_key, in_key, out_key,
                               label_scale=0, resolution=res,
                               tmp_folder=tmp_folder, target='local', max_jobs=8, max_threads=8,
                               assignment_path=ass_path, assignment_key=ass_key)


if __name__ == '__main__':
    convert_cremi(True)
