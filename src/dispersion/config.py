"""provides functions for reading and writing the configuration file."""
#import sys
import os
from warnings import warn
from dispersion.io import (read_yaml_file, read_yaml_string,
                                          write_yaml_file)
from typing import Dict, Type, Union

if os.name == 'nt':
    PLATFORM = "Windows"
elif os.name == 'posix':
    PLATFORM = 'Linux'
else:
    raise OSError("curret os type could not be determined."+
                  " Configuration file location unknown.")


def validate_config(config: Dict[str, Union[str, bool, Dict[str, bool], Dict[str, Union[float, str]]]]) -> None:
    """
    validates the data types of the configuration

    Parameters
    config: dict
        dictionary containing all configuration
    """
    check_type(config['Path'],str)
    if not os.path.isdir(config['Path']):
        raise IOError("directory path for database file system is invalid:" +
                      " <{}>".format(config['Path']))
    check_type(config['Interactive'],bool)
    assert "Modules" in config
    for value in config['Modules'].values():
        check_type(value,bool)
    assert "ReferenceSpectrum" in config
    ref_spec = config['ReferenceSpectrum']
    check_type(ref_spec['Value'],float)
    check_type(ref_spec['SpectrumType'],str)
    check_type(ref_spec['Unit'],str)

def check_type(value: Union[bool, str, float], val_type: Union[Type[bool], Type[str], Type[float]]) -> None:
    """
    checks if value isinstance of val_type, if not raise exception
    """
    if not isinstance(value,val_type):
        raise ValueError("config file data {}".format(value) +
                         "must be of type {}".format(val_type) +
                         ", not {}".format(type(value)))


def default_config():
    """provides default values for the configuration.

    Returns
    -------
    dict or OrderedDict
        the tree of configuration data
    """
    yaml_str = """\
    Path: {}
    File: catalogue.csv
    Interactive: false #for jupyter interfactive editing
    Modules: # which databases to include
      UserData: true
      RefractiveIndexInfo: true
    ReferenceSpectrum: # evaluate n and k at given spectral value
      Value: 632.8
      SpectrumType: wavelength
      Unit: nanometer
    """.format(_get_user_config_dir())
    config = read_yaml_string(yaml_str)
    return config

def _get_user_config_dir():
    if PLATFORM == 'Windows':
        user_dir = os.environ["LOCALAPPDATA"]
        config_dir = os.path.join(user_dir, "dispersion")
    elif PLATFORM == 'Linux':
        home_dir = os.environ["HOME"]
        config_dir = os.path.join(home_dir, ".config",
                                  "dispersion")
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    return config_dir

def _get_package_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path

def _get_config_dir():
    user_dir = _get_user_config_dir()
    file_path = os.path.join(user_dir, 'config.yaml')
    if os.path.isfile(file_path):
        return user_dir
    pkg_dir = _get_package_dir()
    file_path = os.path.join(pkg_dir, 'config.yaml')
    if os.path.isfile(file_path):
        return pkg_dir
    return user_dir

def write_config(config):
    """write the configuration data to file.

    Parameters
    ----------
    config: dict or OrderedDict
        the configuration data

    Warnings
    --------
    No check is made if the config is valid or complete.
    """
    dir_path = _get_config_dir()
    file_path = os.path.join(dir_path, 'config.yaml')
    write_yaml_file(file_path, config)

def read_config():
    """read the configuration data from file.

    Returns
    -------
    dict or OrderedDict
        the configuration data
    """
    dir_path = _get_config_dir()
    file_path = os.path.join(dir_path, 'config.yaml')
    config = read_yaml_file(file_path)
    return config

def get_config():
    """get the configuration data.

    attempt to read the configuration from file, if not found then return
    default values.

    Returns
    -------
    dict or OrderedDict
        the configuration data
    """
    try:
        config = read_config()
    except OSError as exc:
        warn("No configuration file found, generating default config.")
        config = default_config()
    return config

if __name__ == "__main__":
    print(get_config())
