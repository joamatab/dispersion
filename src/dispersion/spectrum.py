"""spectrum is an array of data which defines a physical spectrum

A Spectrum is a numpy array with an associated type which describes
the physical quantity (e.g. wavelength, energy, etc.) and a unit
for specifiying the SI unit of the spectrum (e.g. meter, electronVolt)
"""
import numpy as np
import scipy.constants as constants
from numpy import float64, ndarray
from typing import List, Tuple, Union

def safe_inverse(values: Union[float, ndarray]) -> Union[float, ndarray]:
    '''
    returns the inverse of a scalar or array while
    converting values of 0 to inf and values of inf to 0
    '''
    if isinstance(values,np.ndarray):
        inverse = safe_inverse_array(values)
    else:
        inverse = safe_inverse_scalar(values)
    return inverse

def safe_inverse_array(values: ndarray) -> ndarray:
    '''
    takes the inverse of an array while converting values of 0 to inf
    and values of inf to 0
    '''
    inverse = np.array(values)
    zero_ind = np.isclose(values,0.0)
    inf_ind = np.isclose(values,np.inf)
    safe_inv_ind = ~ (zero_ind | inf_ind)
    inverse[zero_ind] = np.inf
    inverse[inf_ind] = 0.0
    inverse[safe_inv_ind] = 1.0/values[safe_inv_ind]
    return inverse

def safe_inverse_scalar(value: float) -> float:
    '''
    takes the inverse of a scalar while converting values of 0 to inf
    and values of inf to 0
    '''
    if value == 0.0:
        return np.inf
    elif value == np.inf:
        return 0.0
    else:
        return 1.0/value

def frequency_to_standard(unit: str, values: ndarray) -> ndarray:
    """converts frequency values to standardised representation"""
    if not unit == 'hz':
        raise NotImplementedError("support for frequency units other " +
                                  "than Hz not implemented")
    inverse = safe_inverse(values)
    converted_values = constants.c*inverse
    return converted_values


def energy_to_standard(unit: str, values: Union[float, ndarray]) -> Union[float, ndarray]:
    """converts energy values to standardised representation"""
    if not unit == 'ev':
        raise NotImplementedError("support for energy units other " +
                                  "than eV not implemented")
    inverse = safe_inverse(values)
    converted_values = (constants.h*constants.c/(constants.e))*inverse
    return converted_values


def ang_freq_to_standard(unit: str, values: ndarray) -> ndarray:
    """converts angular frequency values to standardised representation"""
    if not unit == '1/s':
        raise NotImplementedError("support for angular frequency " +
                                  "units other than 1/s " +
                                  "not implemented")
    inverse = safe_inverse(values)
    converted_values = 2*np.pi*constants.c*inverse
    return converted_values


def wavenumber_to_standard(unit: str, values: ndarray) -> ndarray:
    """converts wavenumber values to standardised representation"""
    if not unit == '1/cm':
        raise NotImplementedError("support for wavenumber units " +
                                  "other than 1/cm not implemented")
    inverse = safe_inverse(values)
    converted_values = 2*np.pi*inverse*1e-2
    return converted_values


def wavelength_to_standard(unit: str, values: Union[float, ndarray]) -> Union[float, ndarray]:
    """converts wavelength values to standardised representation"""
    if unit == 'nm':
        converted_values = values*1e-9
    elif unit == 'um':
        converted_values = values*1e-6
    else:
        converted_values = values
    return converted_values

def to_wavelength(unit: str, values: Union[float, ndarray]) -> Union[float64, float, ndarray]:
    """converts values from standard to wavelength representation"""
    if unit == 'm':
        converted_values = values
    if unit == 'nm':
        converted_values = np.round(values*1e9, decimals=12)
    elif unit == 'um':
        converted_values = np.round(values*1e6, decimals=9)
    return converted_values

def to_frequency(unit: str, values: ndarray) -> ndarray:
    """converts values from standard to frequency representation"""
    if not unit == 'hz':
        raise NotImplementedError("support for frequency units " +
                                  "other than Hz not implemented")
    inverse = safe_inverse(values)
    return constants.c * inverse

def to_energy(unit: str, values: Union[float, ndarray]) -> Union[float64, ndarray]:
    """converts values from standard to energy representation"""
    if not unit == 'ev':
        raise NotImplementedError("support for energy units " +
                                  " other than eV not implemented")
    inverse = safe_inverse(values)
    return np.round((constants.h*constants.c/(constants.e))*inverse, decimals=9)

def to_ang_freq(unit: str, values: ndarray) -> ndarray:
    """converts values from standard to angular frequency representation"""
    if not unit == '1/s':
        raise NotImplementedError("support for angular frequency units " +
                                  "other than 1/s not implemented")
    inverse = safe_inverse(values)
    return 2*np.pi*constants.c*inverse

def to_wavenumber(unit: str, values: ndarray) -> ndarray:
    """converts values from standard to wavenumber representation"""
    if not unit == '1/cm':
        raise NotImplementedError("support for wavenumber units" +
                                  " other than 1/cm not implemented")
    inverse = safe_inverse(values*1e-2)
    return 2*np.pi*inverse

class Spectrum(object):
    '''
    Class for converting between wavelength/frequency/energy
    when defining spectral quantities
    '''

    SPECTRUM_TYPES = {'wavelength': {'m', 'um', 'nm'},
                      'frequency': {'hz'},
                      'energy': {'ev'},
                      'angularfrequency': {'1/s'},
                      'wavenumber': {'1/cm'}}

    UNIT_TYPES = {'m': {'meter', 'metre', 'm'},
                  'um': {'micrometer', 'micrometre', 'um'},
                  'nm': {'nanometer', 'nanometre', 'nm'},
                  'hz': {'hertz', 'hz'},
                  'ev': {'electronvolt', 'ev'},
                  '1/s': {'1/s', 'rad/s'},
                  '1/cm': {'1/cm'}}

    DISPLAY_NAMES = {'m': 'm',
                     'um': 'um',
                     'nm': 'nm',
                     'hz': 'Hz',
                     'ev': 'eV',
                     '1/s': '1/s',
                     '1/cm': '1/cm'}

    def __init__(self, values: Union[float, Tuple[float64, float64], Tuple[int, float], List[float], List[int]], spectrum_type: str='wavelength', unit: str='m') -> None:
        self.spectrum_type = spectrum_type.lower()
        if self.spectrum_type not in Spectrum.SPECTRUM_TYPES:
            spectral_types = list(Spectrum.SPECTRUM_TYPES.keys())
            raise ValueError("spectrum type [{}] ".format(spectrum_type) +
                             "not valid, valid types are " +
                             "{}".format(spectral_types))
        self.unit = self.standardise_unit(self.spectrum_type, unit.lower())
        if isinstance(values, (list, tuple)):
            values = np.array(values)
        if not ( isinstance(values, np.ndarray) or
                 isinstance(values, float)):
            raise ValueError("values must be array like or float")

        self.values = values
        self.standard_rep = None  # values in standardised representation
        self.standardise_values(values)

    @staticmethod
    def standardise_unit(spectrum_type: str, unit: str) -> str:
        """takes a spectrum_type and unit string and returns them in a
        standardised form"""
        valid = False
        unit_type = ""
        for unit_type, aliases in Spectrum.UNIT_TYPES.items():
            if (unit in aliases and
                    unit_type in Spectrum.SPECTRUM_TYPES[spectrum_type]):
                valid = True
                break
        if valid is False:
            s_types = Spectrum.SPECTRUM_TYPES
            raise ValueError("unit [{}] invalid ".format(unit) +
                             "valid units for type {} ".format(spectrum_type) +
                             "are {}".format(s_types[spectrum_type]))
        return unit_type

    def get_type_unit_string(self):
        """formats the unit and spectrum_type for display as an axis label"""
        unit_str = self.display_string(self.unit)
        type_str = self.spectrum_type
        return "{} ({})".format(type_str, unit_str)

    def contains(self, other: object) -> bool:
        """detects if the values of another Spectrum object are
        completely contained within the range of the values defined
        in this Spectrum object"""
        other_values = other.convert_to(self.spectrum_type, self.unit)
        other_min = np.min(other_values)
        other_max = np.max(other_values)
        self_min = np.min(self.values)
        self_max = np.max(self.values)
        out_of_range = None
        if other_min < self_min:
            out_of_range = other_min
        elif other_max > self_max:
            out_of_range = other_max
        if out_of_range is not None:
            dsu = self.display_string(self.unit)  # display string unit
            raise ValueError("evaluation requested for " +
                             "{} {} {}, ".format(self.spectrum_type,
                                                 out_of_range,
                                                 dsu) +
                             "but valid range is {} {} < {} < {} {}".format(
                                 self_min, dsu, self.spectrum_type,
                                 self_max, dsu))

        return True

    @staticmethod
    def check_type(type_to_check, is_type):
        """test if two spectrum_type strings are equal taking aliases
        into account"""
        test_str = type_to_check.lower()
        if test_str not in Spectrum.SPECTRUM_TYPES:
            raise ValueError("spectrum type " +
                             "{} not valid, valid types are: {}".format(
                                 test_str, Spectrum.SPECTRUM_TYPES.keys()))
        return test_str == is_type

    @staticmethod
    def check_unit(unit, is_unit):
        """test if two unit strings are equal taking aliases
        into account"""
        test_str = unit.lower()
        return test_str == is_unit

    @staticmethod
    def display_string(unit):
        """gets the SI standard form of a unit string for printing"""
        return Spectrum.DISPLAY_NAMES[unit]

    def standardise_values(self, values: Union[float, ndarray]) -> None:
        """check if we need to convert the value in order to obtain
        the wavelength/meter representation"""
        if not (self.spectrum_type == 'wavelength' and
                self.unit == 'm'):
            self.standard_rep = self.convert_from(self.spectrum_type,
                                                  self.unit, values)
        else:
            self.standard_rep = values

    @staticmethod
    def convert_from(spectrum_type: str, unit: str, values: Union[float, ndarray]) -> Union[float, ndarray]:
        """
        converts units from given type into wavelength/meter representation
        """
        if spectrum_type not in Spectrum.SPECTRUM_TYPES:
            raise ValueError("invalid spectrum type {}".format(spectrum_type))

        convert_methods = {'frequency': frequency_to_standard,
                           'energy': energy_to_standard,
                           'angularfrequency': ang_freq_to_standard,
                           'wavenumber': wavenumber_to_standard,
                           'wavelength': wavelength_to_standard}

        converted_values = convert_methods[spectrum_type](unit, values)
        return converted_values

    def convert_to(self, spectrum_type: str, unit: str, in_place: bool=False) -> Union[float64, float, ndarray]:
        """
        converts units from wavelength/meter representation to given type/unit
        """
        spectrum_type = spectrum_type.lower()
        if spectrum_type not in Spectrum.SPECTRUM_TYPES:
            raise ValueError("invalid spectrum type {}".format(spectrum_type))

        unit = Spectrum.standardise_unit(spectrum_type, unit)
        values = self.standard_rep

        convert_methods = {'frequency': to_frequency,
                           'energy': to_energy,
                           'angularfrequency': to_ang_freq,
                           'wavenumber': to_wavenumber,
                           'wavelength': to_wavelength}

        converted_values = convert_methods[spectrum_type](unit, values)
        if in_place:
            self.values = converted_values
            self.spectrum_type = spectrum_type
            self.unit = unit
            return None
        else:
            return converted_values
