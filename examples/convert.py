from paintera_tools import convert_to_paintera_format
from paintera_tools import set_default_shebang


def convert_cremi():
    path = '/g/kreshuk/data/cremi/example/sampleA.n5'
    raw_key = 'raw'
    ass_key = 'node_labels'
    in_key = 'segmentation/watershed'
    out_key = 'paintera'

    shebang = '#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/cluster_env37/bin/python'
    set_default_shebang(shebang)

    res = [40, 4, 4]
    tmp_folder = './tmp_convert'
    convert_to_paintera_format(path, raw_key, in_key, out_key,
                               label_scale=0, resolution=res,
                               tmp_folder=tmp_folder, target='local', max_jobs=8, max_threads=8,
                               assignment_path=path, assignment_key=ass_key)


if __name__ == '__main__':
    convert_cremi()
