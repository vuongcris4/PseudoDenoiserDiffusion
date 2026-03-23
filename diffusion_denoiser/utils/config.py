"""Lightweight Config class replacing mmcv.Config.

Supports:
- Loading Python config files via exec()
- Dot-notation access (cfg.model.num_classes)
- Nested dict mering via merge_from_dict()
- get() with defaults
"""

import os
import os.path as osp
import copy
import tempfile
from pathlib import Path


class ConfigDict(dict):
    """A dictionary that supports attribute-style access."""

    def __getattr__(self, name):
        try:
            val = self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")
        return val

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def copy(self):
        return ConfigDict(super().copy())

    def __deepcopy__(self, memo):
        return ConfigDict(copy.deepcopy(dict(self), memo))


def _dict_to_configdict(d):
    """Recursively convert nested dicts to ConfigDict."""
    if isinstance(d, dict):
        return ConfigDict({k: _dict_to_configdict(v) for k, v in d.items()})
    elif isinstance(d, (list, tuple)):
        return type(d)(_dict_to_configdict(v) for v in d)
    return d


def _merge_a_into_b(a, b):
    """Merge dict a into dict b (b is modified in-place)."""
    for k, v in a.items():
        if k in b and isinstance(b[k], dict) and isinstance(v, dict):
            _merge_a_into_b(v, b[k])
        else:
            b[k] = _dict_to_configdict(v)


class Config:
    """Simple config class that loads Python config files.

    Supports _base_ inheritance like mmcv.Config.

    Usage:
        cfg = Config.fromfile('configs/model.py')
        print(cfg.model.num_classes)
    """

    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = {}
        self._cfg_dict = _dict_to_configdict(cfg_dict)

    @staticmethod
    def fromfile(filepath):
        filepath = osp.abspath(osp.expanduser(filepath))
        if not osp.isfile(filepath):
            raise FileNotFoundError(f'Config file not found: {filepath}')

        cfg_dict = Config._file2dict(filepath)
        return Config(cfg_dict)

    @staticmethod
    def _file2dict(filepath):
        filepath = osp.abspath(filepath)
        cfg_dir = osp.dirname(filepath)

        # Read and exec the Python config
        with open(filepath) as f:
            content = f.read()

        # Execute in a temp module namespace
        cfg_dict = {}
        # We need to handle _base_ first
        temp_globals = {'__file__': filepath, '__name__': '__config__'}
        exec(compile(content, filepath, 'exec'), temp_globals)

        # Extract config variables (skip builtins and dunder)
        for k, v in temp_globals.items():
            if not k.startswith('__') and not callable(v) and k != '__builtins__':
                cfg_dict[k] = v

        # Handle _base_ inheritance
        if '_base_' in cfg_dict:
            base_files = cfg_dict.pop('_base_')
            if isinstance(base_files, str):
                base_files = [base_files]

            base_cfg = {}
            for bf in base_files:
                bf_path = osp.join(cfg_dir, bf) if not osp.isabs(bf) else bf
                base_d = Config._file2dict(bf_path)
                _merge_a_into_b(base_d, base_cfg)

            # Current file overrides base
            _merge_a_into_b(cfg_dict, base_cfg)
            cfg_dict = base_cfg

        return cfg_dict

    def __getattr__(self, name):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return getattr(self._cfg_dict, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._cfg_dict[name] = value

    def get(self, key, default=None):
        return self._cfg_dict.get(key, default)

    def merge_from_dict(self, d):
        """Merge a flat or nested dict. Supports dot-notation keys."""
        expanded = {}
        for k, v in d.items():
            keys = k.split('.')
            curr = expanded
            for subk in keys[:-1]:
                curr = curr.setdefault(subk, {})
            curr[keys[-1]] = v
        _merge_a_into_b(_dict_to_configdict(expanded), self._cfg_dict)

    def __repr__(self):
        return f'Config({dict(self._cfg_dict)})'

    def __contains__(self, key):
        return key in self._cfg_dict

    def __iter__(self):
        return iter(self._cfg_dict)

    def __len__(self):
        return len(self._cfg_dict)
