from paintera_tools import interactive_splitter
from paintera_tools import set_default_shebang


def interactive_splitting():
    shebang = '#! /home/pape/Work/software/conda/miniconda3/envs/main/bin/python'
    set_default_shebang(shebang)

    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    key = 'paintera'
    aff_key = 'predictions/full_affs'

    tmp_folder = 'tmp_split'

    interactive_splitter(path, key, path, aff_key,
                         tmp_folder, 'local', 4, 4)


if __name__ == '__main__':
    interactive_splitting()
