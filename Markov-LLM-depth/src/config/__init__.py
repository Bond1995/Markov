from . import base
from . import sparse
from . import markov

CONFIG_FORMAT_TO_MODULE_MAP = {
    "base": base,
    "sparse": sparse,
    "markov": markov
}


def parse_args_with_format(format, base_parser, args, namespace):
    return CONFIG_FORMAT_TO_MODULE_MAP[format].parse_args(base_parser, args, namespace)


def registered_formats():
    return CONFIG_FORMAT_TO_MODULE_MAP.keys()
