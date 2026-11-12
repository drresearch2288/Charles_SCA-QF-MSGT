"""Loads the YAML configuration (configs/default.yaml) with attribute access."""
import copy
import yaml


class Cfg(dict):
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


def load_config(path="configs/default.yaml", overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:          # "model.quantum_block=none"
        key, val = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return Cfg(copy.deepcopy(cfg))
