from paintera_tools import convert_to_paintera_format
from paintera_tools import set_default_shebang


def convert_cremi():
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    raw_key = 'raw'
    ass_key = 'node_labels'
    in_key = 'segmentation/watershed'
    out_key = 'paintera'

    shebang = '#! /home/pape/Work/software/conda/miniconda3/envs/main/bin/python'
    set_default_shebang(shebang)

    res = [40, 4, 4]
    tmp_folder = './tmp_convert'
    convert_to_paintera_format(path, raw_key, in_key, out_key,
                               label_scale=0, resolution=res,
                               tmp_folder=tmp_folder, target='local', max_jobs=4, max_threads=4,
                               assignment_path=path, assignment_key=ass_key)


if __name__ == '__main__':
    convert_cremi()
