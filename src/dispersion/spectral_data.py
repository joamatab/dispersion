"""describes spectrally dependent data

spectral_data implements the abstract base class SpectralData which
defines a material parameter which has a spectral dependence (e.g.
refractive index, permittivity). Each of the subclasses must implement
the evaluate method which returns the material parameter for a given
Spectrum object.

Classes
-------
SpectralData
    abstract base class
Constant: SpectralData
    for values that are independent of the spectrum
Interpolation: SpectralData
    for tabulated data values
Model: SpectralData
    abstract base class for values generated from a particular model
Sellmeier: Model
    implements the Sellmeier model for refractive index
Sellmeier2: Model
    implements the modified Sellmeier model for refractive index
Polynomial: Model
    implements a polynomial model for refractive index
RefractiveIndexInfo: Model
    implements the RefractiveIndexInfo model for refractive index
Cauchy: Model
    implements the Cauchy model for refractive index
Gases: Model
    implements the Gas model for refractive index
Herzberger: Model
    implements the Herzberger model for refractive index
Retro: Model
    implements the Retro model for refractive index
Exotic: Model
    implements the Exotic model for refractive index
Drude: Model
    implements the Drude model for complex permittivity
DrudeLorentz: Model
    implements the Drude-Lorentz model for complex permittivity
TaucLorentz: Model
    implements the Tauc-Lorentz model for complex permittivity

Notes
-----
for more information on models see https://refractiveindex.info/about
"""
import re
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from dispersion.spectrum import Spectrum
from dispersion.io import _numeric_to_string_table


class SpectralData():
    '''
    Base class for defining a quantity (e.g. refactive index)
    which is defined over a given spectrum (see class Spectrum).
    '''

    def __init__(self, valid_range,
                 spectrum_type='wavelength',
                 unit='nm'):
        self.spectrum_type = spectrum_type
        self.unit = unit
        self.valid_range = Spectrum(valid_range,
                                    spectrum_type=spectrum_type,
                                    unit=unit)

    def suggest_spectrum(self):
        """for plotting the spectral data we take a geometrically
        spaced set of values"""
        lower_bound = np.min(self.valid_range.values)
        upper_bound = np.max(self.valid_range.values)
        suggest = np.geomspace(lower_bound, upper_bound, num=1000)
        return Spectrum(suggest,
                        spectrum_type=self.spectrum_type,
                        unit=self.unit)

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        raise NotImplementedError("This method should be overridden by"+
                                  " a subclass")

class Constant(SpectralData):
    """for spectral data values that are independent of the spectrum"""

    def __init__(self, constant, valid_range=(0, np.inf),
                 spectrum_type='wavelength',
                 unit='m'):
        super(Constant, self).__init__(valid_range,
                                       spectrum_type=spectrum_type,
                                       unit=unit)
        self.constant = constant

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        self.valid_range.contains(spectrum)
        if isinstance(spectrum.values, (list, tuple, np.ndarray)):
            return self.constant * np.ones(len(spectrum.values))
        return self.constant

    def dict_repr(self):
        """
        return a dictionary representation of the object
        """
        data = {
            'DataType': "constant",
            'ValidRange': _numeric_to_string_table(self.valid_range.values),
        }

        data['SpectrumType'] = self.spectrum_type
        data['Unit'] = self.unit
        data['Value'] = self.constant
        return data

class Extrapolation(SpectralData):
    """
    for extending spectral data outside of the valid range.
    Use with caution
    """
    def __init__(self, spectral_data, extended_spectrum,
                 spline_order=2):
        self.base_spectral_data = spectral_data
        self.spline_order = spline_order
        extrap_spectrum = self.get_extrap_spectrum(extended_spectrum)
        min_range = np.min(extrap_spectrum.values)
        max_range = np.max(extrap_spectrum.values)
        spectrum_type = self.base_spectral_data.spectrum_type
        unit = self.base_spectral_data.unit
        super(Extrapolation, self).__init__((min_range, max_range),
                                            spectrum_type=spectrum_type,
                                            unit=unit)
        self.extrapolate_data()

    def get_extrap_spectrum(self,extended_spectrum):
        """
        takes a Spectrum object with one or two values possibly lying outside
        the base spectral range. Raises an error if the values do not lie
        outside the base spectral range. returns a new length two spectrum
        that gives the lower and upper bound for an extrapolation
        """
        base_spectrum = self.base_spectral_data.valid_range
        extended_spectrum.convert_to(spectrum_type= base_spectrum.spectrum_type,
                                     unit= base_spectrum.unit,
                                     in_place= True)
        new_range = np.array(base_spectrum.values)
        if isinstance(extended_spectrum.values, (list, tuple, np.ndarray)):
            # extrapolation both upper and lower
            extrap_values = extended_spectrum.values
            if extrap_values.size > 2:
                raise ValueError(
                    (
                        "extrapolation spectrum may contain at most"
                        + f"2 values not {extrap_values.size}"
                    )
                )

            for extrap_val in extrap_values:
                new_range = self.validate_extrap_val(extrap_val, new_range)
        else:
            # upper or lower
            extrap_val = extended_spectrum.values
            new_range = self.validate_extrap_val(extrap_val, new_range)

        return Spectrum(new_range,
                        spectrum_type= base_spectrum.spectrum_type,
                        unit= base_spectrum.unit)

    def validate_extrap_val(self,extrap_val,base_range):
        """
        checks if extrap_val lies outside base_range and replaces the relevant
        value in base_range with extrap_val.
        """
        if extrap_val < base_range[0]:
            base_range[0] = extrap_val
        elif extrap_val > base_range[1]:
            base_range[1] = extrap_val
        else:
            raise ValueError(
                (
                    (
                        (
                            ("extrapolation value of " + f"{extrap_val} ")
                            + "lies inside the defined range "
                        )
                        + f"{base_range}"
                    )
                    + " therefore extrapolation is not necessary"
                )
            )

        return base_range


    def extrapolate_data(self):
        """makes a spline base on the base data for future lookup"""
        spectrum = self.base_spectral_data.suggest_spectrum()
        evaluation = self.base_spectral_data.evaluate(spectrum)
        self.extrapolation = splrep(spectrum.values, evaluation,
                                    k=self.spline_order)

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the fiven spectrum"""
        try:
            self.base_spectral_data.valid_range.contains(spectrum)
            return self.base_spectral_data.evaluate(spectrum)
        except ValueError as e:
            spectrum.convert_to(self.spectrum_type, self.unit, in_place=True)
            self.valid_range.contains(spectrum)
            return splev(spectrum.values,self.extrapolation)


class Interpolation(SpectralData):
    """for spectral data values that are from tabulated data"""

    def __init__(self, data, spectrum_type='wavelength',
                 unit='m', interp_order=1):
        self.data = data
        self.interp_order = interp_order
        min_range = np.min(data[:, 0])
        max_range = np.max(data[:, 0])
        super(Interpolation, self).__init__((min_range, max_range),
                                            spectrum_type=spectrum_type,
                                            unit=unit)
        self.interpolate_data()

    def interpolate_data(self):
        """interpolates the data for future lookup"""
        self.interpolation = interp1d(self.data[:, 0], self.data[:, 1],
                                      kind=self.interp_order)

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the fiven spectrum"""

        self.valid_range.contains(spectrum)
        values = spectrum.convert_to(self.spectrum_type, self.unit)
        return self.interpolation(values)

    def dict_repr(self):
        """
        return a dictionary representation of the object
        """
        data = {
            'DataType': "tabulated",
            'ValidRange': _numeric_to_string_table(self.valid_range.values),
        }

        data['SpectrumType'] = self.spectrum_type
        data['Unit'] = self.unit
        data['Data'] = _numeric_to_string_table(self.data)
        return data

class Model(SpectralData):
    """for spectral data values depending on model parameters"""

    def __init__(self, model_parameters, valid_range, spectrum_type='wavelength',
                 unit='m'):
        self.model_parameters = model_parameters
        super(Model, self).__init__(valid_range,
                                    spectrum_type=spectrum_type,
                                    unit=unit)
        self.required_spectrum_type = None # set in subclass
        self.required_unit = None # set in subclass
        self.output = None # set in subclass

    def validate_spectrum_type(self):
        tmp_spectrum = Spectrum(1.0,
                                spectrum_type=self.spectrum_type,
                                unit=self.unit)

        if tmp_spectrum.spectrum_type != self.required_spectrum_type:
            raise ValueError(
                (
                    ("spectrum_type for model " + f"<{type(self).__name__}> must")
                    + f" be {self.required_spectrum_type}"
                )
            )


        if tmp_spectrum.unit != self.required_unit:
            raise ValueError(
                (
                    ("unit for model " + f"<{type(self).__name__}> must")
                    + f"be {self.required_unit}"
                )
            )

    def dict_repr(self):
        """
        return a dictionary representation of the object
        """
        data = {
            'DataType': f"model {type(self).__name__}",
            'ValidRange': _numeric_to_string_table(self.valid_range.values),
        }

        data['SpectrumType'] = self.spectrum_type
        data['Unit'] = self.unit
        data['Yields'] = self.output
        data['Parameters'] = _numeric_to_string_table(self.model_parameters)
        return data

    def input_output(self):
        """defines the required inputs and the output spectrum type"""
        raise NotImplementedError("this abstract method needs to be"+
                                  "overridden by a subclass")

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        raise NotImplementedError("this abstract method needs to be"+
                                  "overridden by a subclass")

    def preprocess(self, spectrum):
        """
        check range of spectrum, convert to correct sType and unit and return
        an object with the same tensor order (scalar|vector) with values set
        to 1.0
        """
        self.valid_range.contains(spectrum)
        new_spectrum = spectrum.convert_to(self.spectrum_type,
                                           self.unit)
        if isinstance(spectrum.values, (list, tuple, np.ndarray)):
            ones = np.ones(new_spectrum.shape)
        else:
            ones = 1.0

        return ones, new_spectrum

class Sellmeier(Model):

    '''
    requires wavelength input in micrometers
    returns real part of refractive index only
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(Sellmeier, self).__init__(model_parameters, valid_range,
                                        spectrum_type=spectrum_type,
                                        unit=unit)
        self.required_spectrum_type = 'wavelength'
        self.required_unit = 'um'
        self.output = 'n'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, wavelengths] = self.preprocess(spectrum)
        rhs = self.model_parameters[0]*ones
        wvlsq = np.power(wavelengths, 2)
        for iterc in range(len(self.model_parameters[1::2])):
            cupper = self.model_parameters[iterc*2+1]
            clower = self.model_parameters[iterc*2+2]
            rhs += cupper*wvlsq/(wvlsq-clower**2)
        return np.sqrt(rhs+1.0)

class Sellmeier2(Model):

    '''
    requires wavelength input in micrometers
    returns real part of refractive index only
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(Sellmeier2, self).__init__(model_parameters, valid_range,
                                         spectrum_type=spectrum_type,
                                         unit=unit)
        self.required_spectrum_type = 'wavelength'
        self.required_unit = 'um'
        self.output = 'n'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, wavelengths] = self.preprocess(spectrum)
        rhs = self.model_parameters[0]*ones
        wvlsq = np.power(wavelengths, 2)
        for iterc in range(len(self.model_parameters[1::2])):
            cupper = self.model_parameters[iterc*2+1]
            clower = self.model_parameters[iterc*2+2]
            rhs += cupper*wvlsq/(wvlsq-clower)
        return np.sqrt(rhs+1.0)

class Polynomial(Model):

    '''
    requires wavelength input in micrometers
    returns real part of refractive index only
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(Polynomial, self).__init__(model_parameters, valid_range,
                                         spectrum_type=spectrum_type,
                                         unit=unit)
        self.required_spectrum_type = 'wavelength'
        self.required_unit = 'um'
        self.output = 'n'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, wavelengths] = self.preprocess(spectrum)
        rhs = self.model_parameters[0]*ones

        for iterc in range(len(self.model_parameters[1::2])):
            c_multi = self.model_parameters[iterc*2+1]
            c_power = self.model_parameters[iterc*2+2]
            rhs += c_multi*np.power(wavelengths, c_power)
        return np.sqrt(rhs)

class RefractiveIndexInfo(Model):

    '''
    requires wavelength input in micrometers
    returns real part of refractive index only
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(RefractiveIndexInfo, self).__init__(model_parameters, valid_range,
                                                  spectrum_type=spectrum_type,
                                                  unit=unit)
        self.required_spectrum_type = 'wavelength'
        self.required_unit = 'um'
        self.output = 'n'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, wavelengths] = self.preprocess(spectrum)
        rhs = self.model_parameters[0]*ones
        wvlsq = np.power(wavelengths, 2)
        for iterc in range(len(self.model_parameters[1:8:4])):
            c_multi_upper = self.model_parameters[iterc*4+1]
            c_power_upper = self.model_parameters[iterc*4+2]
            c_multi_lower = self.model_parameters[iterc*4+3]
            c_power_lower = self.model_parameters[iterc*4+4]
            rhs += (c_multi_upper*np.power(wavelengths, c_power_upper)/
                    (wvlsq-np.power(c_multi_lower, c_power_lower)))
        for iterc in range(len(self.model_parameters[9::2])):
            c_multi = self.model_parameters[iterc*2+9]
            c_power = self.model_parameters[iterc*2+10]
            rhs += c_multi*np.power(wavelengths, c_power)
        return np.sqrt(rhs)



class Cauchy(Model):

    '''
    requires wavelength input in micrometers
    returns real part of refractive index only
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(Cauchy, self).__init__(model_parameters, valid_range,
                                     spectrum_type=spectrum_type,
                                     unit=unit)
        self.required_spectrum_type = 'wavelength'
        self.required_unit = 'um'
        self.output = 'n'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, wavelengths] = self.preprocess(spectrum)
        rhs = self.model_parameters[0]*ones
        for iterc in range(len(self.model_parameters[1::2])):
            c_multi = self.model_parameters[iterc*2+1]
            c_power = self.model_parameters[iterc*2+2]
            rhs += c_multi*np.power(wavelengths, c_power)
        return rhs

class Gases(Model):

    '''
    requires wavelength input in micrometers
    returns real part of refractive index only
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(Gases, self).__init__(model_parameters, valid_range,
                                    spectrum_type=spectrum_type,
                                    unit=unit)
        self.required_spectrum_type = 'wavelength'
        self.required_unit = 'um'
        self.output = 'n'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, wavelengths] = self.preprocess(spectrum)
        rhs = self.model_parameters[0]*ones
        wvlinvsq = np.power(wavelengths, -2)
        for iterc in range(len(self.model_parameters[1::2])):
            cupper = self.model_parameters[iterc*2+1]
            clower = self.model_parameters[iterc*2+2]
            rhs += cupper/(clower-wvlinvsq)
        return rhs+1.0

class Herzberger(Model):

    '''
    requires wavelength input in micrometers
    returns real part of refractive index only
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(Herzberger, self).__init__(model_parameters, valid_range,
                                         spectrum_type=spectrum_type,
                                         unit=unit)
        self.required_spectrum_type = 'wavelength'
        self.required_unit = 'um'
        self.output = 'n'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, wavelengths] = self.preprocess(spectrum)
        rhs = self.model_parameters[0]*ones
        wvlsq = np.power(wavelengths, 2)
        rhs += self.model_parameters[1]*np.power(wvlsq-0.028, -1)
        rhs += self.model_parameters[2]*np.power(wvlsq-0.028, -2)
        rhs += self.model_parameters[3]*wvlsq
        rhs += self.model_parameters[4]*np.power(wavelengths, 4)
        rhs += self.model_parameters[5]*np.power(wavelengths, 6)

        return rhs

class Retro(Model):

    '''
    requires wavelength input in micrometers
    returns real part of refractive index only
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(Retro, self).__init__(model_parameters, valid_range,
                                    spectrum_type=spectrum_type,
                                    unit=unit)
        self.required_spectrum_type = 'wavelength'
        self.required_unit = 'um'
        self.output = 'n'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, wavelengths] = self.preprocess(spectrum)
        rhs = self.model_parameters[0]*ones
        wvlsq = np.power(wavelengths, 2)
        rhs += self.model_parameters[1]*wvlsq/(wvlsq-self.model_parameters[2])
        rhs += self.model_parameters[3]*wvlsq

        tmp_p = -2*rhs/(1-rhs)
        tmp_q = -1/(1-rhs)

        return -0.5*tmp_p + np.sqrt(np.power(0.5*tmp_p, 2) - tmp_q)

class Exotic(Model):

    '''
    requires wavelength input in micrometers
    returns real part of refractive index only
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(Exotic, self).__init__(model_parameters, valid_range,
                                     spectrum_type=spectrum_type,
                                     unit=unit)
        self.required_spectrum_type = 'wavelength'
        self.required_unit = 'um'
        self.output = 'n'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, wavelengths] = self.preprocess(spectrum)
        rhs = self.model_parameters[0]*ones
        wvlsq = np.power(wavelengths, 2)

        rhs += self.model_parameters[1]*wvlsq/(wvlsq - self.model_parameters[2])
        rhs += (self.model_parameters[3]*(wavelengths -self.model_parameters[4])/
                (np.power(wavelengths - self.model_parameters[4], 2) +
                 self.model_parameters[5]))
        return np.sqrt(rhs)

class Drude(Model):

    '''
    requires energy input in eV
    returns real and imaginary parts of permittivity
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(Drude, self).__init__(model_parameters, valid_range,
                                    spectrum_type=spectrum_type,
                                    unit=unit)
        self.required_spectrum_type = 'energy'
        self.required_unit = 'ev'
        self.output = 'eps'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, energies] = self.preprocess(spectrum)
        omega_p = self.model_parameters[0] # plasma frequency in eV
        loss = self.model_parameters[1] # loss in eV
        return ones - omega_p**2/(np.power(energies, 2)+1j*loss*energies)

class DrudeLorentz(Model):

    '''
    requires energy input in eV
    returns real and imaginary parts of permittivity
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(DrudeLorentz, self).__init__(model_parameters, valid_range,
                                           spectrum_type=spectrum_type,
                                           unit=unit)
        self.required_spectrum_type = 'energy'
        self.required_unit = 'ev'
        self.output = 'eps'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, energies] = self.preprocess(spectrum)
        omega_p = self.model_parameters[0] # plasma frequency in eV
        pol_str = self.model_parameters[1] # pole strength (0.<#<1.)
        w_res = self.model_parameters[2] # frequency of Lorentz pole in eV
        loss = self.model_parameters[3] # loss in eV
        return ones +  np.conj(pol_str*omega_p**2/((w_res**2-np.power(energies, 2))+1j*loss*energies))

class TaucLorentz(Model):
    '''
    requires energy input in eV
    returns real and imaginary parts of permittivity
    '''
    def __init__(self, model_parameters, valid_range,
                 spectrum_type='wavelength', unit='m'):
        super(TaucLorentz, self).__init__(model_parameters, valid_range,
                                           spectrum_type=spectrum_type,
                                           unit=unit)
        self.required_spectrum_type = 'energy'
        self.required_unit = 'ev'
        self.output = 'eps'
        self.validate_spectrum_type()

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, energies] = self.preprocess(spectrum)
        A = self.model_parameters[0] # oscillator strength
        E0 = self.model_parameters[1] # pole energy
        C = self.model_parameters[2] # pole broadening
        Eg = self.model_parameters[3] # optical bandgap energy
        eps_inf = self.model_parameters[4] # high frequency limit of the real part of permittivity
        eps_imag = self._calc_eps_imag(ones, energies, A, E0, C, Eg)
        eps_real = self._calc_eps_real(energies, A, E0, C, Eg, eps_inf)
        return eps_real + 1j*eps_imag

    def _calc_eps_imag(self, ones, energies, A, E0, C, Eg):
        eps_imag = ones
        E = energies[energies>=Eg]
        eps_imag[energies>=Eg] = ( (1./E) *A*E0*C*(E-Eg)**2 /
                                   ((E**2-E0**2)**2+C**2*E**2))
        eps_imag[energies<Eg] = 0.0
        return eps_imag

    def _calc_eps_real(self, energies, A, E0, C, Eg, eps_inf):
        E = energies
        alpha_ln = ((Eg**2-E0**2)*E**2 +
                    Eg**2*C**2 -
                    E0**2*(E0**2+3*Eg**2))
        alpha_atan = (E**2-E0**2)*(E0**2+Eg**2) + Eg**2*C**2
        alpha = np.sqrt(4*E0**2-C**2)
        gamma = np.sqrt(E0**2 -0.5*C**2)
        zeta4 = (E**2-gamma**2)**2 + 0.25*alpha**2*C**2

        part1 = (0.5*(A*C*alpha_ln/(np.pi*zeta4*alpha*E0)) *
                 np.log( (E0**2+Eg**2+alpha*Eg)/(E0**2+Eg**2-alpha*Eg)))

        part2 = ((-1*A*alpha_atan/(np.pi*zeta4*E0)) *
                 (np.pi - np.arctan( (2*Eg+alpha)/C) +
                  np.arctan( (-2*Eg+alpha)/C)))

        # From original paper, seems to be wrong
        # part3 = ((2.*A*E0*C/(np.pi*zeta4)) *
        #          (Eg*(E**2-gamma**2)*
        #           (np.pi+2*np.arctan2((gamma**2-Eg**2),(alpha*C)))))

        part3_1 = (4.*A*E0/(np.pi*zeta4*alpha))
        part3_2 = Eg*(E**2-gamma**2)
        part3_3 = ( np.arctan2(alpha+2*Eg,C) +
                    np.arctan2(alpha-2*Eg,C))
        part3 = part3_1 * part3_2 * part3_3

        part4 = ((-A*E0*C/(np.pi*zeta4))*
                 ((E**2+Eg**2)/E) *
                 np.log( np.abs(E-Eg)/(E+Eg)))

        part5 = ((2.*A*E0*C*Eg/(np.pi*zeta4)) *
                 np.log( (np.abs(E-Eg)*(E+Eg))/
                         np.sqrt((E0**2-Eg**2)**2+Eg**2*C**2)))

        return eps_inf + part1 + part2 + part3 + part4 +part5



class Fano(Model):

    '''
    this model can be applied to scattering cross sections
    requires energy input in eV
    returns real and imaginary parts of scattering cross section
    '''
    def input_output(self):
        """defines the required inputs and the output spectrum type"""
        self.required_spectrum_type = 'energy'
        self.required_unit = 'ev'
        self.output = 'scattering_cross_setion'

    def evaluate(self, spectrum):
        """returns the value of the spectral data for the given spectrum"""
        [ones, energies] = self.preprocess(spectrum)
        q = self.model_parameters[0] # Fano parameter
        e_r = self.model_parameters[1] # resonant energy in eV
        gamma = self.model_parameters[2] # loss in eV
        epsilon =  2*(energies-e_r)/gamma
        #loss = self.model_parameters[1] # loss in eV
        norm = (1+q**2)
        return (1/norm) *(epsilon+q)**2 / (epsilon**2+1)
