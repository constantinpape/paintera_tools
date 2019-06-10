from paintera_tools import serialize_from_commit
from paintera_tools import set_default_shebang


def serialize_cremi():
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    in_key = 'paintera'
    out_key = 'segmentation/corrected'

    shebang = '#! /home/pape/Work/software/conda/miniconda3/envs/main/bin/python'
    set_default_shebang(shebang)

    tmp_folder = 'tmp_serialize'
    serialize_from_commit(path, in_key, path, out_key,
                          tmp_folder, 4, 'local',
                          relabel_output=True)


if __name__ == '__main__':
    serialize_cremi()
