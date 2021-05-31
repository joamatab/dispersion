"""holds the complex refactive index or permittivity data

material_data implements the Material class which can hold different
representations of spectral data (e.g. refractive index or permittivity).
The data is in the form of either a constant value, tabulated data or a model.
These different representations can be combined e.g. model for n (real part
of refractive index) and constant value for k (imaginary part of refractive
index)

Functions
---------
_check_table_shape
    validate that a numpy array has a given shape

Classes
-------
Material
    processes and interfaces refractive index data.
"""
from __future__ import print_function
import codecs
import numpy as np
#from dispersion import _str_to_class
import dispersion.spectral_data as spectral_data
from dispersion import Spectrum
from dispersion import Constant, Interpolation, \
                                                    Extrapolation

#from dispersion.spectral_data import _numeric_to_string_table
from dispersion.io import (Reader, _numeric_to_string_table,
                           _str_table_to_numeric)
from dispersion.spectral_data import Drude
from dispersion.spectrum import Spectrum
from numpy import complex128
from typing import Any, Dict, List, Optional, Set, Type, Union









def _check_table_shape(table, ncols, name):
    """
    check that numpy array shape has the correct number of columns
    """
    if (not len(table.shape) == 2 or
        not table.shape[1] == ncols):
        raise ValueError("tabulated {} data ".format(name) +
                         "must have shape Nx{}".format(ncols))

class Material():
    '''
    Class for processing refractive index and permittivity data

    Parameters
    ----------
    file_path: str
        file path from which to load data
    fixed_n: float
        fixed real part of refractive index
    fixed_nk: complex
        fixed complex refractive index
    fixed_eps_r: float
        fixed real part of permittivity
    fixed_eps: complex
        fixed complex permittivity
    tabulated_n: Nx2 array
        table of real part of refractive index to interpolate
    tabulated_nk: Nx3 array
        table of real and imaginary refractive index values to interpolate
    tabulated_eps: Nx3 array
        table of real and imaginary permittivity values to interpolate
    model_kw: dict
        model parameters
    spectrum_type: str
        sets the default spectrum type
    unit: str
        sets the default unit
    meta_data: dict
        contains the meta data for the material
    data: dict
        holds one or two SpectralData objects to describe the data
    options: dict
        holds options for the material object
    defaults: dict
        default values for spectrum data

    Warnings
    --------
    the parameters file_path, fixed_n, fixed_nk, fixed_eps_r, fixed_eps,
    tabulated_n, tabulated_nk, tabulated_eps and model_kw are mututally
    exclusive.

    '''

    def __init__(self, **kwargs) -> None:
        #parsing arguments
        parsed_args = self._parse_args(kwargs)
        #set inputs and defaults
        file_path = parsed_args["file_path"]
        #self.meta_data = None
        self.meta_data = {}
        self.meta_data['Reference'] = ""
        self.meta_data['Comment'] = ""
        self.meta_data['Name'] = ""
        self.meta_data['FullName'] = ""
        self.meta_data['Author'] = ""
        self.meta_data['Alias'] = ""
        self.meta_data['MetaComment'] = ""
        self.meta_data['Specification'] = {}
        self._file_data = None
        self.data = {'name': "",
                     'real': None,
                     'imag': None,
                     'complex':None}
        self.options = {'interp_oder':parsed_args["interp_order"]}
        self.defaults = {'unit':parsed_args["unit"],
                         'spectrum_type':parsed_args["spectrum_type"]}

        #process input arguments
        if file_path is not None:
            reader = Reader(file_path)
            file_data = reader.read_file()
            self._process_file_data(file_data)
        elif parsed_args['model_kw'] is not None:
            self._process_model_dict(parsed_args['model_kw'])
        elif parsed_args['tabulated_nk'] is not None:
            self._process_table(parsed_args['tabulated_nk'], 'nk')
        elif parsed_args['tabulated_n'] is not None:
            self._process_table(parsed_args['tabulated_n'], 'n')
        elif parsed_args['tabulated_eps'] is not None:
            self._process_table(parsed_args['tabulated_eps'], 'eps')
        else:
            self._process_fixed_value(parsed_args)

        self._complete_partial_data()

    def _parse_args(self, args: Dict[str, Union[str, Dict[str, Union[str, List[float]]], float]]) -> Dict[str, Optional[Union[int, str, float, Dict[str, Union[str, List[float]]]]]]:
        """
        validated the dictionary of class inputs
        """
        mutually_exclusive = {"file_path", "fixed_n", "fixed_nk",
                              "fixed_eps_r", "fixed_eps",
                              "tabulated_nk", "tabulated_n",
                              "tabulated_eps",
                              "model_kw"}
        inputs = {}
        n_mutually_exclusive = 0
        for arg in args.keys():
            if arg in args and args[arg] is not None:
                if arg in mutually_exclusive:
                    n_mutually_exclusive += 1
            inputs[arg] = args[arg]

        if n_mutually_exclusive == 0:
            raise ValueError("At least one of the following" +
                             " inputs is required: "+
                             "{}".format(mutually_exclusive))
        elif n_mutually_exclusive > 1:
            raise ValueError("Only one of the following" +
                             "inputs is allowed: "+
                             "{}".format(mutually_exclusive))
        # Check types
        str_args = {'file_path', 'spectrum_type', 'unit'}
        str_types = {str}
        self._check_type(inputs, str_args, str_types)
        if inputs['spectrum_type'] is None:
            inputs['spectrum_type'] = 'wavelength'
        if inputs['unit'] is None:
            inputs['unit'] = 'nanometer'
        if 'interp_order' not in inputs:
            inputs['interp_order'] = 1
        # pylint: disable=no-member
        # bug in pylint does not recognise numpy data types
        int_args = {'interp_oder'}
        int_types = {int}
        self._check_type(inputs, int_args, int_types)

        float_args = {"fixed_n", "fixed_eps_r"}
        float_types = {float, np.double}
        self._check_type(inputs, float_args, float_types)

        complex_args = {"fixed_nk", "fixed_eps"}
        complex_types = {complex, np.cdouble}
        self._check_type(inputs, complex_args, complex_types)

        dict_args = {'model_kw'}
        dict_types = {dict}
        self._check_type(inputs, dict_args, dict_types)

        array_args = {'tabulated_nk', 'tabulated_n', 'tabulated_eps'}
        array_types = {np.ndarray}
        self._check_type(inputs, array_args, array_types)
        if inputs['tabulated_nk'] is not None:
            _check_table_shape(inputs['tabulated_nk'], 3, 'nk')
        if inputs['tabulated_n'] is not None:
            _check_table_shape(inputs['tabulated_n'], 2, 'n')
        if inputs['tabulated_eps'] is not None:
            _check_table_shape(inputs['tabulated_eps'], 3, 'eps')

        return inputs

    @staticmethod
    def _check_type(args: Dict[str, Optional[Union[int, str, float, Dict[str, Union[str, List[float]]]]]], names: Set[str], types: Any) -> None:
        """
        raises TypeError if the names keys in args dict are not in the
        set of types. If name is not in args, place a default value of None.
        """
        for arg in names:
            if arg in args and args[arg] is not None:
                invalid_type = False
                for _type in types:
                    if isinstance(args[arg], _type):
                        invalid_type = True
                if invalid_type is False:
                    raise TypeError("argument " +
                                    "{} must be".format(arg) +
                                    " of types: {}".format(types))
            else:
                args[arg] = None


    def _complete_partial_data(self) -> None:
        """
        if only partial data was provided then set remaining parameters
        to constant value of 0.
        """
        if self.data['real'] is None:
            self.data['real'] = Constant(0.0)
        if self.data['imag'] is None:
            self.data['imag'] = Constant(0.0)

    def remove_absorption(self):
        """
        sets loss (k or epsi) to constant zero value

        Warnings
        --------
        has no effect if the material is defined as via complex data instead of
        separate real and imaginary parts.
        """
        self.data['imag'] = Constant(0.0)

    def extrapolate(self, new_spectrum, spline_order=2):
        """extrpolates the material data

        extrapolates the material data to cover the range defined by the
        spectrum new_spectrum. if new_spectrum has only one element, the data
        will be extrapolated from the relevant end of its valid range up to the
        value given by new_spectrum. spline_order defines the order of the
        spline used for extrapolation. The results of the extrapolation depend
        heavily on the order chosen, so please check the end result to make
        sure it make physical sense.

        Parameters
        ----------
        new_spectrum: Spectrum
            the values to exrapolate to
        spline_order: int
            the order of spline to use for interpolation -> extrpolation

        Raises
        ------
        NotImplementedError
            if the material is defined as via a complex value

        """

        if self.data['complex'] is None:
            for data_name in ['real', 'imag']:
                if isinstance(self.data[data_name], Constant):
                    continue
                self.data[data_name] = Extrapolation(self.data[data_name],
                                                     new_spectrum,
                                                     spline_order=spline_order)
        else:
            raise NotImplementedError("extrapolation not implemented " +
                                      "for materials with real and imaginary "+
                                      "parts not independent from each other")

    def _process_fixed_value(self, inputs: Dict[str, Optional[Union[float, str, int]]]) -> None:
        '''use fixed value inputs to set n/k or permittivity

        the fixed value is converted to a SpectralData.Constant object and
        included in the data dict

        Parameters
        ----------
        inputs: dict
            the dict holding the fixed value
        '''
        if inputs['fixed_n'] is not None:
            self.data['name'] = 'nk'
            self.data['real'] = Constant(inputs['fixed_n'])
            #self._k = Constant(0.0)
        elif inputs['fixed_nk'] is not None:
            self.data['name'] = 'nk'
            self.data['real'] = Constant(np.real(inputs['fixed_nk']))
            self.data['imag'] = Constant(np.imag(inputs['fixed_nk']))
        elif inputs['fixed_eps_r'] is not None:
            self.data['name'] = 'eps'
            self.data['real'] = Constant(inputs['fixed_eps_r'])
            #self._epsi = Constant(0.0)
        elif inputs['fixed_eps'] is not None:
            self.data['name'] = 'eps'
            self.data['real'] = Constant(np.real(inputs['fixed_eps']))
            self.data['imag'] = Constant(np.imag(inputs['fixed_eps']))
        else:
            raise RuntimeError("Failed to set a constant value for n,k or eps")

    def _process_model_dict(self, model_dict: Dict[str, Union[str, List[float]]]) -> None:
        """use model parameter input to set n/k or permittivity

        use model_dict to return a SpectralData.Model object and sets the
        relevant n/k or permittivity class attributes

        Parameters
        ----------
        model_dict: dict
            contains data for model creates (see notes)

        Raises
        ------
        ValueError
            if the model output does not yield n/k or permittivity

        Notes
        -----
        model_dict must contain the fields:
        name: str
            class name of the model (see spectral_data.py)
        spectrum_type: str
            spectrum Type (see spctrum.py)
        unit: str
            spetrum unit (see spctrum.py)
        valid_range: 2x1 np.array
            min and max of the spectral range for which the model is valid
        parameters: np.array
            all paramters (i.e. coefficients) needed for the model
        """
        model_class = self._str_to_class(model_dict['name'])
        #model_class = MODELS[model_dict['name']]
        kws = {}
        if "spectrum_type" in model_dict:
            kws['spectrum_type'] = model_dict['spectrum_type']
            self.defaults['spectrum_type'] = model_dict['spectrum_type']
        if "unit" in model_dict:
            kws['unit'] = model_dict['unit']
            self.defaults['unit'] = model_dict['unit']
        model = model_class(model_dict['parameters'],
                            model_dict['valid_range'], **kws)
        if model.output == 'n':
            self.data['name'] = 'nk'
            self.data['real'] = model
        elif model.output == 'k':
            self.data['name'] = 'nk'
            self.data['imag'] = model
        elif model.output == 'nk':
            self.data['name'] = 'nk'
            self.data['complex'] = model
        elif model.output == 'epsr':
            self.data['name'] = 'eps'
            self.data['real'] = model
        elif model.output == 'epsi':
            self.data['name'] = 'eps'
            self.data['imag'] = model
        elif model.output == 'eps':
            self.data['name'] = 'eps'
            self.data['complex'] = model
        else:
            raise ValueError("model output <{}> invalid".format(model.output))

    @staticmethod
    def _str_to_class(field: str) -> Type[Drude]:
        """evaluates string as a class.

        tries to evaluate the given string as a class from the spectral_data
        module.

        Parameters
        ----------
        field: str
            name to convert to class

        Raises
        ------
        NameError
            the given field is not an attribute
        TypeError
            the given field is an attribute but not a class
        """
        try:
            identifier = getattr(spectral_data, field)
        except AttributeError:
            raise NameError("%s doesn't exist." % field)
        if isinstance(identifier, type):
            return identifier
        raise TypeError("%s is not a class." % field)

    def _process_file_data(self, file_dict):
        """set meta_data and data from dictionary"""
        self._file_data = file_dict
        self.meta_data = {}
        self.meta_data['Reference'] = file_dict['MetaData']['Reference']
        self.meta_data['Comment'] = file_dict['MetaData']['Comment']
        self.meta_data['Name'] = file_dict['MetaData']['Name']
        self.meta_data['FullName'] = file_dict['MetaData']['FullName']
        self.meta_data['Author'] = file_dict['MetaData']['Author']
        self.meta_data['MetaComment'] = file_dict['MetaData']['MetaComment']
        self.meta_data['Specification'] = file_dict['MetaData']['Specification']
        datasets = file_dict['Datasets']
        #self.dataTypes = []
        #self.dataSets = []
        for dataset in datasets:
            data_type, identifier = dataset['DataType'].split()
            #meta_data = dataset['MetaData']
            if data_type == 'tabulated':
                #data is tabulated
                dataset['Data'] = _str_table_to_numeric(dataset.pop('Data'))
                self._process_table(dataset['Data'], identifier,
                                    meta_data=dataset)
            elif data_type in {'formula', 'model'}:
                #data is a formula with coefficients
                self._process_formula_data(dataset)
            else:
                raise ValueError("data type {} not supported".format(data_type))

    def _process_table(self, table, identifier, meta_data=None):
        """
        Uses a table(np.ndarray) and metadata to set relevant class attributes.
        """
        if meta_data is None:
            meta_data = {}

        if ('SpectrumType' in meta_data and \
            meta_data['SpectrumType'] and \
            meta_data['SpectrumType'] is not None):
            self.defaults['spectrum_type'] = meta_data['SpectrumType']
        if ('Unit' in meta_data and \
            meta_data['Unit'] and \
            meta_data['Unit'] is not None):
            self.defaults['unit'] = meta_data['Unit']

        if identifier == 'nk':
            self.data['name'] = 'nk'
            self.data['real'] = self._spec_data_from_table(table[:, [0, 1]])
            self.data['imag'] = self._spec_data_from_table(table[:, [0, 2]])
        elif identifier == 'n':
            self.data['name'] = 'nk'
            self.data['real'] = self._spec_data_from_table(table)
        elif identifier == 'k':
            self.data['name'] = 'nk'
            self.data['imag'] = self._spec_data_from_table(table)
        elif identifier == 'eps':
            self.data['name'] = 'eps'
            self.data['real'] = self._spec_data_from_table(table[:, [0, 1]])
            self.data['imag'] = self._spec_data_from_table(table[:, [0, 2]])


    def _spec_data_from_table(self, data):
        '''
        Convert table to SpectralData object.

        Parameters
        ----------
        data: Nx2 np.array
            the tabulated spectral data

        Returns
        -------
        Constant(SpectralData)
            if tabulated data has 1 row the data is constant
        Interpolation(SpectralData)
            interpolation of the tabulated data
        '''
        n_rows = data.shape[0]
        spec_type = self.defaults['spectrum_type']
        unit = self.defaults['unit']
        if n_rows == 1:
            return Constant(data[0, 1],
                            valid_range=(data[0, 0], data[0, 0]),
                            spectrum_type=spec_type, unit=unit)

        return Interpolation(data, spectrum_type=spec_type,
                             unit=unit)

    def _process_formula_data(self, data_dict):
        '''prepare dictionary of data for processing.

        create model_dict and call process_model_dict use range and coefficients
        in input dictionary to return a SpectralData.Model
        '''
        model_dict = {}
        meta_data = data_dict
        data_type, identifier = meta_data['DataType'].split()
        if not (data_type in {'formula', 'model'}):
            raise ValueError("dataType <{}>".format(data_type) +
                             " not a valid formula or model")
        if data_type == 'formula':
            identifier = int(identifier)

        if meta_data['ValidRange']:
            valid_range = meta_data['ValidRange'].split()

        for i_valid_range, v_range in enumerate(valid_range):
            valid_range[i_valid_range] = float(v_range)
        model_dict['valid_range'] = valid_range

        coefficients = data_dict['Data'].split()
        for iter_coeff, coeff in enumerate(coefficients):
            coefficients[iter_coeff] = float(coeff)
        model_dict['parameters'] = np.array(coefficients)


        if meta_data['SpectrumType']:
            model_dict['spectrum_type'] = meta_data['SpectrumType']
        else:
            model_dict['spectrum_type'] = self.defaults['spectrum_type']

        if meta_data['Unit']:
            model_dict['unit'] = meta_data['Unit']
        else:
            model_dict['unit'] = self.defaults['unit']

        method_ids = {1: 'Sellmeier', 2: 'Sellmeier2',
                      3: 'Polynomial', 4: 'RefractiveIndexInfo',
                      5: 'Cauchy', 6: 'Gases',
                      7: 'Herzberger', 8: 'Retro',
                      9: 'Exotic'}

        if isinstance(identifier, int):
            model_dict['name'] = method_ids[identifier]
        else:
            model_dict['name'] = identifier

        self._process_model_dict(model_dict)


    def get_nk_data(self, spectrum: Spectrum,
                    spectrum_type: str='wavelength',
                    unit: str='meter') -> Union[complex, complex128]:
        '''
        return complex refractive index for a given input spectrum.

        Parameters
        ----------
        spectrum: np.array or Spectrum
            the spectral values to evaluate
        spectrum_type: str {'wavelength', 'frequency', 'energy'}
            type of spectrum
        unit: str {'meter', 'nanometer', 'micrometer', 'hertz', 'electronvolt'}
            unit of spectrum (must match spectrum type)

        Returns
        -------
        np.complex128
            the complex n/k values (if input spectrum has size == 1)
        np.array with np.complex128 dtype
            the complex n/k values (if input spectrum has size > 1)
        '''
        if isinstance(spectrum, Spectrum):
            spectrum_values = spectrum.values
            spectrum_type = spectrum.spectrum_type
            unit = spectrum.unit
        else:
            spectrum_values = spectrum

        spectrum = Spectrum(spectrum_values,
                            spectrum_type=spectrum_type,
                            unit=unit)

        if not (self.data['name'] == 'nk' or self.data['name'] == 'eps'):
            raise ValueError("data type {}".format(self.data['name']) +
                             "cannot be converted to refractive index")

        if self.data['complex'] is None:
            real = self.data['real'].evaluate(spectrum)
            imag = 1j*self.data['imag'].evaluate(spectrum)
            complex_val = real+imag
        else:
            complex_val = self.data['complex'].evaluate(spectrum)

        if self.data['name'] == 'eps':
            complex_val = np.sqrt(complex_val)
        return complex_val

    def get_permittivity(self, spectrum_values,
                         spectrum_type='wavelength',
                         unit='meter'):
        '''
        return complex permittivity for a given input spectrum.

        Parameters
        ----------
        spectrum: np.array or Spectrum
            the spectral values to evaluate
        spectrum_type: str {'wavelength', 'frequency', 'energy'}
            type of spectrum
        unit: str {'meter', 'nanometer', 'micrometer', 'hertz', 'electronvolt'}
            unit of spectrum (must match spectrum type)

        Returns
        -------
        np.complex128
            the complex permittivity values (if input spectrum has size == 1)
        np.array with np.complex128 dtype
            the complex permittivity values (if input spectrum has size > 1)
        '''

        if isinstance(spectrum_values, Spectrum):
            spectrum = spectrum_values
        else:
            spectrum = Spectrum(spectrum_values,
                                spectrum_type=spectrum_type,
                                unit=unit)

        if not (self.data['name'] == 'nk' or self.data['name'] == 'eps'):
            raise ValueError("data type {}".format(self.data['name']) +
                             "cannot be converted to refractive index")

        if self.data['complex'] is None:
            real = self.data['real'].evaluate(spectrum)
            imag = 1j*self.data['imag'].evaluate(spectrum)
            complex_val = real+imag
        else:
            complex_val = self.data['complex'].evaluate(spectrum)

        if self.data['name'] == 'nk':
            complex_val = np.power(complex_val, 2)
        return complex_val

    def get_maximum_valid_range(self):
        """find maximum spectral range that spans real and imaginary data.

        Checks both real and imaginary parts of spectral data and finds the
        maximum spectral range which is valid for both parts.

        Returns
        -------
        2x1 np.array
            the maximum valid range
        """
        if not(self.data['name'] == 'nk' or self.data['name'] == 'eps'):
            raise RuntimeError("valid_range cannot be defined as "+
                               "Material does not yet contain "+
                               " a valid n/k or permittivity spectrum")

        if self.data['complex'] is None:
            real_range_std = self.data['real'].valid_range.standard_rep
            imag_range_std = self.data['imag'].valid_range.standard_rep
            real_lower = np.min(real_range_std)
            real_upper = np.max(real_range_std)
            imag_lower = np.min(imag_range_std)
            imag_upper = np.max(imag_range_std)
            lower = np.max([real_lower, imag_lower])
            upper = np.min([real_upper, imag_upper])
        else:
            lower = np.min(self.data['complex'].valid_range.values)
            upper = np.max(self.data['complex'].valid_range.values)
        max_range = np.array([lower, upper])
        spec = Spectrum(max_range)
        return spec.convert_to(self.defaults['spectrum_type'],
                               self.defaults['unit'])

    @staticmethod
    def utf8_to_ascii(string):
        """converts a string from utf8 to ascii"""
        uni_str = codecs.encode(string, 'utf-8')
        ascii_str = codecs.decode(uni_str, 'ascii', 'ignore')
        return ascii_str

    def print_reference(self):
        """print material reference"""
        print(self.utf8_to_ascii(self.meta_data['Reference']))

    def print_comment(self):
        """print material comment"""
        print(self.utf8_to_ascii(self.meta_data['Comment']))


    def plot_nk_data(self, **kwargs):
        """plots the real and imaginary part of the refractive index"""
        self._plot_data('nk', **kwargs)

    def plot_permittivity(self, **kwargs):
        """plots the real and imaginary part of the permittivity"""
        self._plot_data('permittivity', **kwargs)

    def _plot_data(self, data_label, **kwargs):
        """internal function used for plotting spectral data"""
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exp:
            raise ModuleNotFoundError("plotting requires the matplotlib",
                                      " package to be installed")

        plot_data = self._prepare_plot_data(**kwargs)
        if 'axes' not in kwargs:
            plot_data['axes'] = plt.axes()
        else:
            plot_data['axes'] = kwargs['axes']
        if data_label == 'nk':
            data = self.get_nk_data(plot_data['spectrum'])
            labels = ['n', 'k']
        elif data_label == 'permittivity':
            data = self.get_permittivity(plot_data['spectrum'])
            labels = ['eps_r', 'eps_i']

        data_r = np.real(data)
        data_i = np.imag(data)
        # pylint: disable=protected-access
        # this is the only way to access the color cycler
        axes = plot_data['axes']
        spectrum = plot_data['spectrum']
        if spectrum.values.size == 1:
            color = next(axes._get_lines.prop_cycler)['color']
            plt.axhline(data_r, label=labels[0], color=color)
            color = next(axes._get_lines.prop_cycler)['color']
            plt.axhline(data_i, label=labels[1], ls='--', color=color)
        else:
            plt.plot(spectrum.values, data_r, label=labels[0])
            plt.plot(spectrum.values, data_i, ls='--', label=labels[1])
        plt.legend(loc='best')
        plt.ylabel("{}, {}".format(labels[0], labels[1]))
        xlabel = spectrum.get_type_unit_string()
        plt.xlabel(xlabel)

    def _prepare_plot_data(self, **kwargs):
        """internal function to prepare data for plotting"""
        plot_data = {}
        if 'spectrum_type' not in kwargs:
            plot_data['spectrum_type'] = self.defaults['spectrum_type']
        else:
            plot_data['spectrum_type'] = kwargs['spectrum_type']
        if 'unit' not in kwargs:
            plot_data['unit'] = self.defaults['unit']
        else:
            plot_data['unit'] = kwargs['unit']
        if 'values' not in kwargs:
            spectrum = self.get_sample_spectrum()
            values = spectrum.convert_to(plot_data['spectrum_type'],
                                         plot_data['unit'])


        else:
            values = kwargs['values']
            if isinstance(values, (list, tuple)):
                values = np.array(values)
        spectrum = Spectrum(values,
                            spectrum_type=plot_data['spectrum_type'],
                            unit=plot_data['unit'])
        plot_data['spectrum'] = spectrum
        return plot_data

    def get_sample_spectrum(self):
        """spectrum which covers the maximum valid range of the material data"""
        max_range = self.get_maximum_valid_range()
        if max_range[0] == 0.0 or max_range[1] == np.inf:
            values = np.geomspace(100, 2000, 1000)
            spectrum = Spectrum(values, spectrum_type='wavelength',
                                unit='nm')
        else:
            values = np.geomspace(max_range[0], max_range[1], 1000)
            if values[0] < max_range[0]:
                values[0] = max_range[0]
            if values[-1] > max_range[1]:
                values[-1] = max_range[1]
            spectrum = Spectrum(values,
                                spectrum_type=self.defaults['spectrum_type'],
                                unit=self.defaults['unit'])
        return spectrum

    def prepare_file_dict(self):
        #if self._file_data is not None:
        #    return self._file_data
        file_dict = {}
        file_dict['MetaData'] = {}
        #file_dict[]
        for key in self.meta_data:
            if key == "Alias":
                continue
            file_dict['MetaData'][key] = self.meta_data[key]
        file_dict['Datasets'] = self.dataset_to_dict()
        return file_dict

    def add_dtype_suffix(self, dtype, data_part):
        if self.data['name'] == 'nk':
            if data_part == 'real':
                dtype += " n"
            elif data_part == 'imag':
                dtype += " k"
            elif data_part == 'complex':
                dtype += ' nk'
        elif self.data['name'] == 'eps':
            if data_part == 'real':
                dtype += " eps_r"
            elif data_part == 'imag':
                dtype += ' eps_k'
            elif data_part == 'complex':
                dtype += ' eps'
        else:
            raise ValueError("data name could not be parsed")
        return dtype


    def dataset_to_dict(self):
        """
        generate a file_data type dictionary from this object

        Parameters
        ----------

        material_data: dict
            keys: name, real, imag, complex

        Returns
        -------
        dict
            a list of dicts that has a format suitable for writing to file
        """
        datasets = []
        if self.data['complex'] is None:
            data_parts = ['real', 'imag']
        else:
            data_parts = 'complex'


        for data_part in data_parts:
            spec_data = self.data[data_part]
            data_dict = spec_data.dict_repr()
            if isinstance(spec_data, (Constant, Interpolation)):
                dtype = data_dict['DataType']
                dtype = self.add_dtype_suffix(dtype, data_part)
                data_dict['DataType'] = dtype

            datasets.append(data_dict)
        datasets = self.collapse_datasets(datasets)
        return datasets

    def collapse_datasets(self, datasets):
        n_collapsable = 0
        for dataset in datasets:
            if dataset['DataType'] == 'tabulated n':
                n_data = _str_table_to_numeric(dataset['Data'])
                n_collapsable += 1
            if dataset['DataType'] == 'tabulated k':
                k_data = _str_table_to_numeric(dataset['Data'])
                n_collapsable += 1

        if n_collapsable < 2:
            return datasets

        collapse = True
        if not np.all(n_data[:, 0] == k_data[:, 0]):
            collapse = False

        if datasets[0]['Unit'] != datasets[1]['Unit']:
            collapse = False

        if datasets[0]['SpectrumType'] != datasets[1]['SpectrumType']:
            collapse = False

        if not collapse:
            return datasets

        new_dataset = {}
        new_dataset['Unit'] = datasets[0]['Unit']
        new_dataset['SpectrumType'] = datasets[0]['SpectrumType']
        new_dataset['DataType'] = 'tabulated nk'
        k_data = k_data[:, 1].reshape(k_data.shape[0], 1)
        new_data = np.concatenate([n_data, k_data], axis=1)
        new_dataset['Data'] = _numeric_to_string_table(new_data)
        return [new_dataset]

class EffectiveMedium(Material):

    def __init__(self, spectrum, material1, material2, filling_fraction):
        self.data = {'name': "",
                     'real': None,
                     'imag': None,
                     'complex':None}
        self.options = {'interp_oder':1}
        self.defaults = {'unit':'m',
                         'spectrum_type':'wavelength'}
        self.spectrum = spectrum
        self.mat1 = material1
        self.mat2 = material2
        self.frac = filling_fraction
        if self.frac < 0. or self.frac > 1.0:
            raise ValueError("filling fraction must be between "+
                             "0. and 1.")

        self.create_effective_data()

    def create_effective_data(self):
        """
        this must be implemented in a subclass
        """
        raise NotImplementedError("create_effective_data must be defined" +
                                  " in a subclass")

class MaxwellGarnett(EffectiveMedium):

    def create_effective_data(self):
        #def get_maxwell_garnet(eps_base, eps_incl, vol_incl):
        small_number_cutoff = 1e-6
        print(self.frac)
        eps_base = self.mat1.get_permittivity(self.spectrum)
        eps_incl = self.mat2.get_permittivity(self.spectrum)
        factor_up = 2*(1-self.frac)*eps_base+(1+2*self.frac)*eps_incl
        factor_down = (2+self.frac)*eps_base+(1-self.frac)*eps_incl
        if np.any(abs(factor_down)) < small_number_cutoff:
            raise ValueError('effective medium is approximately singular')
        eps_eff = eps_base*factor_up/factor_down
        self.spectrum.convert_to("wavelength", 'm', in_place=True)
        table = np.concatenate([[self.spectrum.values],
                                [np.real(eps_eff)],
                                [np.imag(eps_eff)]]).T
        self._process_table(table, "eps")

class Bruggeman(EffectiveMedium):

    def create_effective_data(self):
        eps_b = self.mat1.get_permittivity(self.spectrum)
        eps_a = self.mat2.get_permittivity(self.spectrum)
        solution1 = self._get_bruggeman_solution1(eps_b, eps_a)
        solution2 = self._get_bruggeman_solution2(eps_b, eps_a)
        self.sol1 = solution1
        self.sol2 = solution2
        indices1 = np.imag(solution1) >= 0.0
        indices2 = np.imag(solution2) >= 0.0
        eps_eff = np.zeros(eps_b.shape, dtype=np.cdouble)
        eps_eff[indices2] = solution2[indices2]
        eps_eff[indices1] = solution1[indices1]
        self.spectrum.convert_to("wavelength", 'm', in_place=True)
        table = np.concatenate([[self.spectrum.values],
                                [np.real(eps_eff)],
                                [np.imag(eps_eff)]]).T
        self._process_table(table, "eps")

    def _get_bruggeman_solution1(self, eps_b, eps_a):
        q = -0.5*eps_a*eps_b
        p = 0.5*(self.frac*(eps_b-2*eps_a) + (1-self.frac)*(eps_a-2*eps_b) )
        return -p*0.5  + np.sqrt((0.5*p)**2 - q)

    def _get_bruggeman_solution2(self, eps_b, eps_a):
        q = -0.5*eps_a*eps_b
        p = 0.5*(self.frac*(eps_b-2*eps_a) + (1-self.frac)*(eps_a-2*eps_b) )
        return -p*0.5  - np.sqrt((0.5*p)**2 - q)







if __name__ == "__main__":
    pass
