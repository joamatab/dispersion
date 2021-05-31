import pytest
import numpy as np
from dispersion import Catalogue
from dispersion import Spectrum
from dispersion import get_config

spectrum = Spectrum(0.5, unit='um')

config = get_config()
# config['Path'] = root_path

def test_mdb_init():
    mdb = Catalogue(config=config)

def test_get_mat():
    mdb = Catalogue(config=config)
    mat = mdb.get_material("Silver")
    nk = mat.get_nk_data(spectrum)
    assert np.isclose(np.real(nk), 0.05)
    assert np.isclose(np.imag(nk), 3.1308839999999996)

if __name__ == "__main__":
    test_get_mat()
