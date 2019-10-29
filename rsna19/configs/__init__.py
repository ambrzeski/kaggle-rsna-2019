import json


def load(config_path):
    dct = json.load(open(config_path))
    config = type('', (), dct)()
    return config
