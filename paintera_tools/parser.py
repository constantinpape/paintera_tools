import os
import json


def parse_merges(attributes):
    if not isinstance(attributes, dict):
        assert os.path.exists(attributes)
        with open(attributes) as f:
            attributes = json.load(f)
    sources = attributes["sourceInfo"]["sources"]
    for source in sources:
        print(list(source.keys()))


if __name__ == '__main__':
    path = ''
    parse_merges(path)
