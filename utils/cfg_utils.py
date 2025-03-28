from types import NoneType

from detectron2.config import CfgNode

_VALID_TYPES = {tuple, list, str, int, float, bool, NoneType}


# TODO convert from dict to CfgNode, move to separate file. Use this to save the config in the pickled model. Then also create tooling to combine existing configs and weights.
def convert_cfg_node_to_dict(cfg_node, key_list: list = []):
    """Convert a config node to dictionary"""
    if not isinstance(cfg_node, CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                "Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_node_to_dict(v, key_list + [k])
        return cfg_dict


def convert_dict_to_cfg_node(cfg_dict):
    """Convert a dictionary to config node"""
    cfg_node = CfgNode(cfg_dict)
    return cfg_node
