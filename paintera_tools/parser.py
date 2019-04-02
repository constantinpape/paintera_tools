import os
import json


def parse_actions(attributes, type_filter):
    if not isinstance(attributes, dict):
        assert os.path.exists(attributes), attributes
        with open(attributes) as f:
            attributes = json.load(f)
    sources = attributes["paintera"]["sourceInfo"]["sources"]
    label_source = None
    for source in sources:
        if source["type"] == "org.janelia.saalfeldlab.paintera.state.LabelSourceState":
            label_source = source["state"]
            break

    assert label_source is not None
    actions = label_source["assignment"]["data"]["actions"]

    filtered_actions = []
    for act in actions:
        if act["type"] in type_filter:
            filtered_actions.append(act)
    return filtered_actions


def parse_merges(attributes):
    actions = parse_actions(attributes, type_filter=("MERGE",))
    frag_ids1, frag_ids2, seg_ids = [], [], []
    for act in actions:
        act = act["data"]
        frag_ids1.append(act['fromFragmentId'])
        frag_ids2.append(act['intoFragmentId'])
        seg_ids.append(act['segmentId'])
    return frag_ids1, frag_ids2, seg_ids
