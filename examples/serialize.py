from paintera_tools import serialize_from_commit


def serialize_cremi():
    path = '/home/pape/Work/data/cremi/example/sampleA.n5'
    in_key = 'paintera'
    out_key = 'segmentation/corrected'

    tmp_folder = 'tmp_serialize'
    serialize_from_commit(path, in_key, path, out_key,
                          tmp_folder, 4, 'local',
                          relabel_output=True)


if __name__ == '__main__':
    serialize_cremi()
