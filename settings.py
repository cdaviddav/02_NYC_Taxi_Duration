# -*- coding: utf-8 -*-
from pathlib import Path

import os
from configparser import RawConfigParser, ConfigParser, ExtendedInterpolation, _UNSET
from datetime import date

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'conf', 'default.conf')


def parse_int_list(s):
    return [int(x) for x in s.split(',')]


def parse_float_list(s):
    return [float(x) for x in s.split(',')]


def parse_string_list(s):
    return [x.strip() for x in s.split(',')]


converters = {
    'intarray': parse_int_list,
    'floatarray': parse_float_list,
    'stringarray': parse_string_list
}


def get_path(path: Path):
    """
    Establishes that a given path exists
    :param path: Pathlib path to get
    :return: the path
    """
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
    return path


class DefaultConfig(ConfigParser):
    def __init__(self, working_dir=None):
        ConfigParser.__init__(self, converters=converters, interpolation=ExtendedInterpolation())
        self.read(CONFIG_FILE_PATH)
        self._cache = {}
        self._working_dir = os.path.dirname(__file__) if working_dir is None else working_dir




class Settings:
    def __init__(self, root_path=None):
        self.root_path = Path(root_path) if root_path is not None else Path(__file__).parent
        self.config = DefaultConfig()

    @property
    def reports(self):
        return get_path(self.root_path / 'reports')

    @property
    def figures(self):
        return get_path(self.reports / 'figures')

    @property
    def csv(self):
        return get_path(self.reports / 'csv')

    @property
    def tensorboard(self):
        return get_path(self.reports / 'tensorboard')

    @property
    def data(self):
        return get_path(self.root_path / "data")

    @property
    def raw_data(self):
        return get_path(self.data / "raw_data")

    @property
    def log_dir(self):
        return get_path(self.root_path / "log")