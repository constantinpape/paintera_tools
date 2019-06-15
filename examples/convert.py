from paintera_tools import convert_to_paintera_format
from paintera_tools import set_default_shebang


def convert_cremi(input_path, with_assignments):
    """ Convert segmentation to paintera format.

    You can donwload the example data from
    https://drive.google.com/open?id=15hZmM4cu_H_ruhlgXilNWgDZWMpuo9XK

    Arguments:
        input_path [str]: path to n5 file with raw data and segmentation
        with_assignments [bool]: convert to format with / without fragment-segment assignment
    """

    # NOTE: the raw data needs to be multiscale, i.e. 'raw_key' needs to be a group with
    # datasets 'raw_key/s0' ... 'raw_key/sN'. It's ok if there's only a single scale, i.e. 'raw_key/s0'
    raw_key = 'volumes/raw'

    # we have two options: create a dataset with or without fragment-segment assignments
    if with_assignments:
        # with assignments: we need an assignment path and key
        # and we use the watersheds as input segmentation
        ass_path = input_path
        ass_key = 'node_labels'
        in_key = 'volumes/segmentation/watershed'
    else:
        # without assignments: no assignment path or key
        # and we use the multicut segmentation as input segmentation
        ass_path = ass_key = ''
        in_key = 'volumes/segmentation/multicut'

    # output key: we stote the new paintera dataset here
    out_key = 'paintera'

    # shebang to environment with all necessary dependencies
    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    set_default_shebang(shebang)

    res = [40, 4, 4]
    tmp_folder = './tmp_convert'
    convert_to_paintera_format(input_path, raw_key, in_key, out_key,
                               label_scale=0, resolution=res,
                               tmp_folder=tmp_folder, target='local', max_jobs=8, max_threads=8,
                               assignment_path=ass_path, assignment_key=ass_key)


if __name__ == '__main__':
    input_path = '/g/kreshuk/data/cremi/example/sampleA.n5'
    convert_cremi(input_path, True)
