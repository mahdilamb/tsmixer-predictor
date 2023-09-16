"""Methods for working with config."""
import argparse

import yaml


def save(args: argparse.Namespace, path: str):
    """Save an argparse Namespace to file."""
    with open(path, "w") as fp:
        yaml.dump(vars(args), fp)
    return args


def load(path: str):
    """Load a config."""
    with open(path, "rb") as fp:
        return yaml.safe_load(fp)
