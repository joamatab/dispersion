"""provides functions for reading and writing files.

implements a set of functions for reading and writing yaml files. Also
implements a reader and writer object for interfacing with material data files.

Functions
---------
validate_table
    checks that tabulated data can be interpolated
fix_table
    removes rows of table which stop interpolation from happening
_str_table_to_numeric
    convert a string table to a numpy array
read_yaml_file
    convert a file with yaml format to dict
read_yaml_string
    convert a string with yaml format to dict
write_yaml_file
    write dict to a yaml format file
print_yaml_string
    print a dict in yaml format to stdout

Classes
-------
Reader
    reads refractive index data from file
Writer
    writes refractive index data to file
"""
import os
import sys
import codecs
import warnings
import re
import pathlib
from collections import OrderedDict
import numpy as np
from typing import Dict, List, Union

USE_RUAMEL = True
try:
    from ruamel.yaml import YAML
    from ruamel.yaml import scalarstring
    from ruamel.yaml.comments import CommentedMap
except ModuleNotFoundError as exc:
    warnings.warn("preferred yaml package ruamel.yaml not installed, falling" +
                  " back to PyYAML, writing yaml files may give inconsistent" +
                  " round trip results")
    USE_RUAMEL = False
    import yaml
#from dispersion.material import _str_table_to_numeric
#USE_RUAMEL = False
#import yaml

def multi_key(dict_obj, key_set):
    for key in key_set:
        if key in dict_obj:
            return dict_obj[key]
    else:
        raise KeyError("None of the keys {}".format(key_set) +
                       " were present")

def _numeric_to_string_table(table):
    """
    returns a string representation of a numpy array
    """
    string_array = np.array2string(table)
    no_brackets = re.sub(r'[\[\]]', '', string_array)
    no_indent = re.sub(r'\n ', r"\n", no_brackets)
    no_indent = no_indent.rstrip().lstrip()
    return no_indent

def _str_table_to_numeric(table):
    '''
    takes tabulated data in string form
    and converts to a numpy array
    '''

    if isinstance(table, np.ndarray):
        numeric_table = table
    elif isinstance(table, str):
        #table is a str
        numeric_table = []
        for row in table.split('\n'):
            if row.isspace() or row == "":
                break
            numeric_col = []
            for col in row.split():
                numeric_col.append(float(col))
            numeric_table.append(numeric_col)
        numeric_table = np.array(numeric_table)

    else:
        raise TypeError("table of type " +
                        "{} cannot be parsed".format(type(table)))
    if validate_table(numeric_table) is False:
        numeric_table = fix_table(numeric_table)
    return numeric_table



def validate_table(tabulated_data):
    '''
    check that spectral part (first column) is
    monotonically increasing to be able to interpolate
    '''
    return np.all(tabulated_data[1:, 0] > tabulated_data[:-1, 0])


def fix_table(tabulated_data):
    '''
    throw out rows which break strict monotonicity
    '''
    n_cols = tabulated_data.shape[1]
    new_rows = [tabulated_data[0, :]]
    last_valid = tabulated_data[0, 0]
    for row in range(1, tabulated_data.shape[0]):
        if not tabulated_data[row, 0] > last_valid:
            continue
        else:
            new_rows.append(tabulated_data[row, :])
            last_valid = tabulated_data[row, 0]
    return np.array(new_rows).reshape(-1, n_cols)

def read_yaml_file(file_path: str):
    """opens yaml file and returns contents as a dict like.

    Parameters
    ----------
    file_path: str
        path to the yaml file

    Returns
    -------
    dict or OrderedDict
        the data from the yaml file

    Notes
    -----
    If USE_RUAMEL is true an OrderedDict is returned, otherwise a dict is
    returned.
    """
    file_path = pathlib.Path(file_path)

    with open(file_path, 'r', encoding="utf-8") as fpt:
        if USE_RUAMEL:
            yaml_obj = YAML()
            yaml_data = yaml_obj.load(fpt)
        else:
            try:
                yaml_data = yaml.load(fpt, Loader=yaml.FullLoader)
            except:
                yaml_data = yaml.safe_load(fpt)
    return yaml_data

def read_yaml_string(string_data):
    """converts a string in yaml format to a dict like.

    Parameters
    ----------
    string_data: str
        string in the yaml format

    Returns
    -------
    dict or OrderedDict
        the data from the yaml file

    Warnings
    --------
    No attempt is made to check if string_data is in correct yaml format.

    Notes
    -----
    If USE_RUAMEL is true an OrderedDict is returned, otherwise a dict is
    returned.
    """
    if USE_RUAMEL:
        yaml_obj = YAML()
        yaml_data = yaml_obj.load(string_data)
    else:
        try:
            yaml_data = yaml.load(string_data, Loader=yaml.FullLoader)
        except:
            yaml_data = yaml.safe_load(string_data)
    return yaml_data

def write_yaml_file(file_path, dict_like):
    """write a dict_like object to a file using the given file path.

    Parameters
    ----------
    file_path: str
        path with file name and extension
    dict_like: dict or OrderedDict
        the data to be written to file
    """
    with open(file_path, 'w', encoding='utf8') as fpt:
        if USE_RUAMEL:
            yaml_obj = YAML()
            yaml_obj.indent(mapping=4, sequence=4, offset=2)
            scalarstring.walk_tree(dict_like)
            yaml_obj.dump(dict_like, fpt)
        else:
            yaml.dump(dict_like, fpt)

def print_yaml_string(dict_like):
    """print a dict_like object to stdout in yaml format.

    Parameters
    ----------
    dict_like: dict or OrderedDict
        the data to be written to printed
    """
    if USE_RUAMEL:
        yaml_obj = YAML()
        yaml_obj.dump(dict_like, sys.stdout)
    else:
        print(yaml.dump(dict_like))

def prepend_text_to_file(file_path, text, extra_line=False):
    """
    read information from a file and rewrite with text prepended
    """
    with open(file_path, 'r', encoding='utf8') as fpt:
        current_text = fpt.read()

    with open(file_path, 'w', encoding='utf8') as fpt:
        text = text.rstrip('\n\r')
        text = text.replace("\n", "\n#")
        text = "#" + text
        text += "\n"
        if extra_line:
            text += "\n"
        fpt.write(text)
        fpt.write(current_text)


class Reader():
    """interface for reading refractive index data from file.

    Attributes
    ----------
    file_path: str
        path to the file to be read
    extension: str
        file type to read
    default_file_dict: dict
        default values to use

    Methods
    -------
    read_file
        read the associated file
    """

    FILE_META_DATA_KEYS = {"Comment", "Reference", "Author",
                           "Name", "FullName", "MetaComment"}

    DATASET_META_DATA_KEYS = {'ValidRange', 'DataType',
                              'SpectrumType', 'Unit',
                              'Yields'}

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        fname, extension = os.path.splitext(file_path)
        self.extension = extension
        self.default_file_dict = self._create_default_file_dict()

    def _create_default_file_dict(self) -> Dict[str, Union[Dict[str, str], List[Dict[str, str]], str]]:
        """default values for the material data."""
        file_dict = {'MetaData': {}}
        for mdk in Reader.FILE_META_DATA_KEYS:
            file_dict['MetaData'][mdk] = ""
        file_dict['Datasets'] = [Reader._create_default_data_dict()]
        file_dict['MetaData']['Specification'] = {}
        #file_dict['MetaComment'] = ""
        file_dict['FilePath'] = self.file_path
        return file_dict

    @staticmethod
    def _create_default_data_dict() -> Dict[str, str]:
        """default values for a data set."""
        dataset_dict = {}
        for mdk in Reader.DATASET_META_DATA_KEYS:
            dataset_dict[mdk] = ""
        dataset_dict['Data'] = []
        return dataset_dict

    def read_file(self):
        """reads the material data file from file.

        Returns
        -------
        dict
            the data from the material file
        """
        txt_types = {'.txt', '.csv'}
        if self.extension in txt_types:
            return self._read_text_file()
        elif self.extension == '.yml':
            return self._read_yaml_mat_file()
        else:
            raise ValueError("extension " +
                             "{} not supported".format(self.extension) +
                             ", supported extensions are (.yml|.csv|.txt)")

    def _read_text_data(self):
        """read data stored in a .txt or .csv file."""
        fname, ext = os.path.splitext(self.file_path)
        try:
            if ext == '.txt':
                try:
                    data = np.loadtxt(self.file_path, encoding='utf-8')
                except TypeError:
                    with open(self.file_path, encoding = 'utf-8') as fpt:
                        data = np.loadtxt(fpt)
            elif ext == '.csv':
                try:
                    data = np.loadtxt(self.file_path, encoding='utf-8',
                                      delimiter=',')
                except TypeError:
                    with open(self.file_path, encoding = 'utf-8') as fpt:
                        data = np.loadtxt(fpt,delimiter=',')
        except IOError as exc:
            raise exc
        data_dict = self._create_default_data_dict()
        data_dict['Data'] = data
        return data_dict

    def _read_text_comment(self):
        '''
        returns text from contiguous lines in file beginning with #.
        '''
        comment = []
        with codecs.open(self.file_path, 'r', 'utf-8') as fpt:
            for line in fpt:
                if line[0] == "#":
                    comment.append(line[1:].rstrip("\n\r"))
                else:
                    return comment
        return comment

    def _read_text_file(self):
        """
        text files (.txt,.csv) may only contain tabulated nk data
        plus metadata. Metadata should be at the beginning of the
        file, one item per line. The line must start with '#' to
        denote metadata. Metadata is written in the key:value
        structure.
        """

        dataset = self._read_text_data()
        comment = self._read_text_comment()
        file_dict = dict(self.default_file_dict)
        file_dict['Datasets'][0] = dataset
        multi_line_comment = ""
        for line in comment:
            kwd_arg = line.split(":")
            if len(kwd_arg) == 1:
                multi_line_comment += line + "\n"
            elif len(kwd_arg) == 2:
                kwd = kwd_arg[0].upper().lstrip()
                arg = kwd_arg[1].rstrip("\n\r").lstrip()
                valid = False
                for key in Reader.FILE_META_DATA_KEYS:
                    if kwd.startswith(key.upper()):
                        if key == 'Specification':
                            file_dict['MetaData'][key] = str(arg)
                        else:
                            file_dict['MetaData'][key] = arg
                        valid = True
                        break
                if valid is False:
                    for key in Reader.DATASET_META_DATA_KEYS:
                        if kwd.startswith(key.upper()):
                            file_dict['Datasets'][0][key] = arg
                            valid = True
                            break
                if valid is False:
                    KeyError("keyword " +
                             "[{}] in comment header invalid".format(kwd))
            else:
                raise RuntimeError(" string \":\" may only appear" +
                                   "once per line in comment header")
        if multi_line_comment != "":
            file_dict['MetaData']['MetaComment'] = multi_line_comment
        dataset = file_dict['Datasets'][0]

        if dataset['DataType'] == "":
            if dataset['Data'].shape[1] == 3:
                dataset['DataType'] = 'tabulated nk'
            elif dataset['Data'].shape[1] == 2:
                dataset['DataType'] = 'tabulated n'
        return file_dict

    def _read_yaml_mat_file(self):
        '''
        The refractiveindex.info database format
        '''
        #yaml_stream = open(self.file_path, 'r', encoding="utf-8")
        #dir_path = os.path.dirname(os.path.realpath(__file__))
        yaml_data = read_yaml_file(self.file_path)

        file_dict = dict(self.default_file_dict)
        file_dict = self._process_mat_dict(file_dict, yaml_data)
        mcomment = "\n".join(self._read_text_comment())
        mcomment = mcomment[:-1]
        file_dict['MetaData']['MetaComment'] = mcomment
        return file_dict

    def _process_mat_dict(self, file_dict, yaml_dict):
        """
        copies raw data from a yaml file to the format used in this package.

        Parameters
        ----------
        file_dict: dict
            dict with the correct field names for creating a materialdata object
        yaml_dict: dict
            dict with the raw data describing a material

        Returns
        -------
        dict
            the updated file_dict
        """
        for kwd in yaml_dict:
            kwd = kwd.upper()
            arg = yaml_dict[kwd]
            valid = False
            if kwd == 'DATA':
                valid = True
                file_dict['Datasets'] = self._process_mat_data_dict(arg)
            elif kwd == 'SPECS':
                valid = True
                file_dict['MetaData']['Specification'] = arg
            if valid is False:
                for key in Reader.FILE_META_DATA_KEYS:
                    if kwd.startswith(key.upper()):
                        file_dict['MetaData'][key] = arg
                        valid = True
                        break
            if valid is False:
                KeyError("keyword [{}] in file invalid".format(kwd))
        return file_dict

    def _process_mat_data_dict(self, mat_data):
        """
        creates a list of dicts to generate spectral data sets

        Parameters
        ----------
        mat_data: list of dicts
            data for creating spectral data sets

        Returns
        -------
        list of dicts
            formated data for use in this package
        """
        aliases = {'ValidRange': {'validRange', 'range',
                                  'spectra_range', 'wavelength_range'},
                   'DataType':'type'}
        dataset_list = []
        for n_data_set, dataset in enumerate(mat_data):
            dataset_list.append(Reader._create_default_data_dict())
            data_dict = dataset_list[-1]
            try:
                data_type = dataset['type'].lstrip()
            except KeyError:
                data_type = dataset['DataType'].lstrip()
            data_dict['DataType'] = data_type
            if (data_type.startswith('formula') or
                    data_type.startswith('model')):
                try:
                    data_dict['Data'] = dataset['coefficients']
                except KeyError:
                    data_dict['Data'] = dataset['Parameters']
            elif data_type.startswith('tabulated'):
                try:
                    data_dict['Data'] = dataset['data']
                except KeyError:
                    data_dict['Data'] = dataset['Data']
            else:
                raise KeyError("data type <{}> invalid".format(data_type))

            for kwd in dataset:
                valid = False
                arg = dataset[kwd]
                kwd = kwd.upper()
                for key in Reader.DATASET_META_DATA_KEYS:
                    if valid:
                        break
                    if key in aliases.keys():
                        all_names = aliases[key]
                    else:
                        all_names = {key}
                    for alias in all_names:
                        if kwd == alias.upper():
                            data_dict[key] = arg
                            valid = True
                            break
                if valid is False:
                    KeyError("keyword <{}> in file invalid".format(kwd))

        return dataset_list


class Writer():
    """interface for writing refractive index data to file.

    Attributes
    ----------
    file_path: str
        path to the file to be written
    extension: str
        file type to write
    file_name: str
        name of the file
    material: MaterialData
        the material object to be written

    Methods
    -------
    write_file
        write the material data to file

    Notes
    -----
    if data_dict is not an OrderedDict, the order in which the data and
    metadata is written cannot be controlled.
    """

    KEY_ALIASES = {"Reference": "REFERENCES",
                   "Author": "AUTHOR",
                   "FullName": "FULLNAME",
                   "Name": "NAME",
                   "Comment": "COMMENTS",
                   "Specification": "SPECS",
                   "Datasets": "DATA"}

    RII_ALIASES = {"ValidRange": "wavelength_range",
                   "DataType": "type",
                   "Parameters": "coefficients",
                   "Data":"data",
                   "Model:Sellmeier":"formula 1",
                   "Model:Sellmeier2":"formula 2",
                   "Model:Polynomial":"formula 3",
                   "Model:RefractiveIndexInfo":"formula 4",
                   "Model:Cauchy":"formula 5",
                   "Model:Gases":"formula 6",
                   "Model:Herzberger":"formula 7",
                   "Model:Retro":"formula 8",
                   "Model:Exotic":"formula 9"}

    DATA_META_DATA_KEYS = ['Reference', 'Author',
                           'FullName', 'Name', 'Comment']

    DATASET_META_DATA_KEYS = ['ValidRange', 'DataType',
                              'SpectrumType', 'Unit']

    def __init__(self, file_path, material):
        self.file_path = file_path
        fname, extension = os.path.splitext(file_path)
        self.extension = extension
        self.file_name = fname
        self.file_dict = material.prepare_file_dict()
        self.ignore_constant = True
        self.use_rii_aliases = False


    def write_file(self, ignore_constant=True,
                   use_rii_aliases=False):
        """
        write the data to the file_path
        """
        self.ignore_constant = ignore_constant
        self.use_rii_aliases = use_rii_aliases
        meta_comment = self.file_dict['MetaData'].pop('MetaComment')
        #raise NotImplementedError("writing material files not yet implemented")
        txt_types = {'.txt', '.csv'}
        if self.extension in txt_types:
            self._write_text_file()
            if not meta_comment == "":
                prepend_text_to_file(self.file_path, meta_comment)
        elif self.extension == '.yml':
            self._write_yaml_file()
            if not meta_comment == "":
                prepend_text_to_file(self.file_path, meta_comment,
                                     extra_line=True)
        else:
            raise ValueError("extension" +
                             "{} not supported".format(self.extension) +
                             ", supported extensions are (.yml|.csv|.txt)")


    def _write_text_file(self):
        """
        writer for .txt and .csv files
        """
        header = ""
        for key in Writer.DATA_META_DATA_KEYS:
            if key in self.file_dict['MetaData']:
                alias = Writer.KEY_ALIASES[key]
                header += "{}: {}\n".format(alias,
                                            self.file_dict['MetaData'][key])


        # for item in Writer.KEY_ALIASES.items():
        #     if item[0] in self.file_dict:
        #         header += "{}: {}\n".format(item[1], self.file_dict[item[0]])
        if len(self.file_dict['Datasets']) < 1:
            raise ValueError("no datasets present in the file dict")
        data_dict = self.file_dict["Datasets"][0]

        spec_type = multi_key(data_dict, {'spectrum_type', 'SpectrumType'})
        header += "SPECTRUMTYPE: {}\n".format(spec_type)
        unit = multi_key(data_dict, {'unit', 'Unit'})
        header += "UNIT: {}\n".format(unit)
        dtype = multi_key(data_dict, {'type', 'DataType'})
        header += "DATATYPE: {}\n".format(dtype)
        # dtype = multi_key(data_dict,{'type','DataType'})
        # if dtype not in {'tabulated_nk', 'tabulated_n'}:
        #     raise NotImplementedError("writing data to text file only"+
        #                               "supported for tabulated_nk and"+
        #                               "tabulted_n data.")

        if self.extension == '.txt':
            delimiter = "\t"
        elif self.extension == '.csv':
            delimiter = ","
        data = multi_key(data_dict, {'data', 'Data'})
        data = _str_table_to_numeric(data)
        np.savetxt(self.file_path, data,
                   delimiter=delimiter,
                   header=header,
                   fmt="%.8f")


    def _write_yaml_file(self):
        """
        writer for .yml files
        """

        pop_keys = []
        for key in self.file_dict:
            if self.file_dict[key] == "":
                pop_keys.append(key)
        for key in pop_keys:
            self.file_dict.pop(key)
        pop_dataset = []

        for ids, dataset in enumerate(self.file_dict['Datasets']):
            if ("constant" in dataset['DataType'] and
                    self.ignore_constant):
                pop_dataset.append(ids)
        for pop_id in pop_dataset:
            self.file_dict['Datasets'].pop(pop_id)
            # if isinstance(dataset['data'], np.ndarray):
            #     string_array = np.array2string(dataset['data'])
            #     no_brackets = re.sub(r'[\[\]]', '', string_array)
            #     no_indent = re.sub(r'\n ', r"\n", no_brackets)
            #     dataset['data'] = no_indent
            #if isinstance(dataset['coefficients'], )

        yaml_dict = {}
        for key in self.file_dict['MetaData']:
            if key in Writer.KEY_ALIASES:
                md = self.file_dict['MetaData'][key]
                yaml_dict[Writer.KEY_ALIASES[key]] = md
            else:
                yaml_dict[key] = md

        yaml_dict['DATA'] = self.file_dict['Datasets']


        if self.use_rii_aliases:
            yaml_datasets = []
            for dataset in yaml_dict['DATA']:

                yaml_dataset = {}
                for key in dataset:
                    if key in {"SpectrumType", "Unit", "Yields"}:
                        continue
                    value = dataset[key]
                    if value in Writer.RII_ALIASES:
                        value = Writer.RII_ALIASES[value]
                    if key in Writer.RII_ALIASES:
                        yaml_dataset[Writer.RII_ALIASES[key]] = value
                    else:
                        yaml_dataset[key] = value
                yaml_datasets.append(yaml_dataset)
            yaml_dict['DATA'] = yaml_datasets
        specs = yaml_dict.pop('SPECS')

        if len(specs) > 0:
            spec_order = ['n_absolute',
                          'wavelength_vacuum',
                          'film_thickness',
                          'substrate',
                          'temperature',
                          'pressure',
                          'deposition_temperature',
                          'direction']
            if USE_RUAMEL:
                yaml_spec = CommentedMap()
            else:
                yaml_spec = {}
            for spec_key in spec_order:
                for key, val in specs.items():
                    if spec_key == key:
                        yaml_spec[spec_key] = val
            yaml_dict['SPECS'] = yaml_spec

        write_yaml_file(self.file_path, yaml_dict)
