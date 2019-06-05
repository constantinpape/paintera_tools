import os
import json
import z5py


# TODO proprly implement this
def parse_actions(project_dir, dataset_name=None):
    attrs = os.path.join(project_dir, 'attributes.json')
    with open(attrs, 'r') as f:
        attrs = json.load(f)

    datasets = attrs['paintera']['sourceInfo']['sources']
    label_datasets = [ds['state'] for ds in datasets if ds['type'].split('.')[-1] == 'LabelSourceState']
    assert len(label_datasets) > 0

    if dataset_name is None:
        assert len(label_datasets) == 1
        dataset = label_datasets[0]
    else:
        dataset_names = [ds['name'] for ds in label_datasets]
        assert dataset_name in dataset_names
        dataset = [ds for ds in label_datasets if ds['name'] == dataset_name][0]

    # load n5 label dataset
    source = dataset['source']['source']['meta']
    # TODO support hdf5 to
    path, key = source['n5'], source['dataset']
    print("N5-file corresponding to dataset:", path, ":", key)

    # load assignments
    assignments = dataset['assignment']
    # TODO handle other types
    ass_type = assignments['type'].split('.')[-1]
    assert ass_type == 'FragmentSegmentAssignmentOnlyLocal'
    assignments = assignments['data']
    actions = assignments['actions']

    # check if we have an inital lut
    initial_lut = assignments['initialLut']['data']['N5']['data']
    lut_path, lut_key = initial_lut['n5'], initial_lut['dataset']
    with z5py.File(lut_path, 'r') as f:
        have_lut = lut_key in f
    # TODO handle initial lut
    assert have_lut, "Don't support LUT"

    # load actions
    # print(actions[:10])

    # load canvas
    # print(list(dataset.keys()))
    print(dataset['source'])
    return actions[:]


def serialize_from_project():
    """ Serialize corrected segmentation from project directory
    """
    raise NotImplementedError("Serializing directly from project is not implemented yet. Use serialize_from_commit.")
