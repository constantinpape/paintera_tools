import z5py


def save_assignments(assignments, save_path, save_key):
    # save the dense assignments in temp folder
    chunks = (min(len(assignments), 1000000), 2)
    f_ass = z5py.File(save_path)
    ds = f_ass.require_dataset(save_key, shape=assignments.shape, chunks=chunks,
                               compression='gzip', dtype='uint64')
    ds[:] = assignments
