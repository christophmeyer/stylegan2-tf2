import numpy as np
import yaml


class ModelConfig():
    def __init__(self, config_path):
        with open(config_path) as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.__dict__ = config_dict


def num_feature_maps_builder(feature_maps_base,
                             feature_maps_decay,
                             feature_maps_min,
                             feature_maps_max):
    def num_feature_maps(n_layer):
        return np.clip(int(feature_maps_base / (2.0 ** (n_layer * feature_maps_decay))), feature_maps_min, feature_maps_max)
    return num_feature_maps
