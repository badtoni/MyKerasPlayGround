import json
from dotmap import DotMap
import os
import time


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    return config


def get_config(folder_path):
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        config_file = os.path.join(folder_path, 'model_config.json')
        config = process_config(config_file)
    except:
        print("missing or invalid arguments")
        exit(0)
    
    return config