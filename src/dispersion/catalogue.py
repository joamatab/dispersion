"""A database for administering refractive index data files

The database set of data files on disk which describe spectrally resolved
refractive index or permittivity data

Functions
---------

rebuild_database
    rebuilds all of the modules in the database

Classes
-------

Catalogue
    administers the catalogue of material data files
"""
from __future__ import print_function
import os
import warnings
import numpy as np
import pandas as pd
import pathlib

ver = pd.__version__
split = ver.split(".")
PANDAS_MINOR_VERSION = int(split[1])

from dispersion.material import Material
from dispersion.spectrum import Spectrum
from dispersion.config import get_config, validate_config
from dispersion.io import read_yaml_file
from typing import Dict, Optional, Union

CONFIG = get_config()


def rebuild_catalogue():
    dirpath = pathlib.Path(CONFIG["Path"])
    catalogue = dirpath / "catalogue.csv"

    if catalogue.exists():
        ans = input("really rebuild and overwrite material catalogue? (y/n) ")
        if ans.upper() == "N":
            return

    cat = Catalogue(rebuild="All")
    cat.save_to_file()


class Catalogue(object):
    """
    administers the set of material data files

    this class is used to find and load data files from disk which describe
    spectrally resolved refractive index or permittivity data into Material
    objects.

    Parameters
    ----------

    config: dict or None
        configuration data
    rebuild: list or None
        which modules which should be rebuilt
    base_path: str
        where the database file structure is stored
    file_name: str
        name of the catalogue file
    database: pandas.DataFrame
        the catalogue of material files
    qgrid_widget: qgrid.widget
        iPython widget for interactive editing of the catalogue
    make_grid: function
        qgrid function for refreshing the interactive interface
    reference_spectrum: Spectrum
        catalogue will provide n/k values at the reference spectrum value
    rii_loader: dict
        temporary dict used in constructing the refractive index info catalogue
    """

    # pylint: disable=no-member
    # bug in pylint does not recognise numpy data types

    META_DATA = {
        "Alias": str,
        "Name": str,
        "FullName": str,
        "Author": str,
        "Comment": str,
        "Reference": str,
        "SpectrumType": str,
        "Unit": str,
        "SpectrumLowerBound": np.double,
        "SpectrumUpperBound": np.double,
        "N_Reference": np.double,
        "K_Reference": np.double,
        "Path": str,
        "Module": str,
    }
    NA_VALUES = {"N_Reference": [""], "K_Reference": [""]}

    def __init__(
        self,
        config: Optional[
            Dict[str, Union[str, bool, Dict[str, bool], Dict[str, Union[float, str]]]]
        ] = None,
        rebuild: str = "None",
    ):
        if config is None:
            config = get_config()
        validate_config(config)
        self.make_reference_spectrum(config)
        self.config = config
        self.base_path = config["Path"]
        self.file_name = config["File"]
        if rebuild == "All":
            df = pd.DataFrame(columns=Catalogue.META_DATA.keys())
        else:
            dtypes = Catalogue.META_DATA
            df = pd.read_csv(
                os.path.join(self.base_path, self.file_name),
                dtype=dtypes,
                na_values=Catalogue.NA_VALUES,
                keep_default_na=False,
            )

        if rebuild != "None":
            df = self.build_catalogue(df, rebuild)
        self.database = df
        self.qgrid_widget = None
        if self.config["Interactive"]:
            try:
                import qgrid
            except ModuleNotFoundError as exc:
                raise Exception(
                    "Interactive mode requires the package" + " qgrid"
                ) from exc
            self.make_qgrid = qgrid.show_grid

    def build_catalogue(self, df, rebuild):
        """
        read specified modules and return a dataframe combining all modules
        """
        config_modules = self.config["Modules"]
        all_modules = {
            "RefractiveIndexInfo": self.read_ri_info_db,
            "Filmetrics": self.read_filmetrics_db,
            "UserData": self.read_user_data_db,
        }
        df_new = pd.DataFrame(columns=Catalogue.META_DATA.keys())
        for module, valid in config_modules.items():
            if valid:
                print("Building {}".format(module))
                if module in rebuild or rebuild == "All":
                    db_path = module
                    dir_path = os.path.join(self.base_path, db_path)
                    read_function = all_modules[module]
                    dframe = read_function(dir_path)
                    if PANDAS_MINOR_VERSION > 22:
                        df_new = df_new.append(dframe, sort=False, ignore_index=True)
                    else:
                        df_new = df_new.append(dframe, ignore_index=True)
                else:
                    dframe = df[df.Database == module]
                    if PANDAS_MINOR_VERSION > 22:
                        df_new = df_new.append(dframe, sort=False, ignore_index=True)
                    else:
                        df_new = df_new.append(dframe, ignore_index=True)

            else:
                if rebuild == module:
                    raise ValueError(
                        "Tried to rebuild module"
                        + "{}".format(rebuild)
                        + ", however this module is disabled"
                        + " in the configuration."
                    )
        return df_new

    def view_interactive(self):
        """
        returns a read only qgrid instance for interactively viewing the
        catalogue
        """
        if self.config["Interactive"] is False:
            raise ValueError("interactivity disabled in config")
        grid_options = {"editable": False}
        self.qgrid_widget = self.make_qgrid(
            self.database, show_toolbar=True, precision=3, grid_options=grid_options
        )
        return self.qgrid_widget

    def edit_interactive(self):
        """
        returns an editable qgrid instance for interactively viewing the
        catalogue. Call the method save_interactive to save any changes made
        """
        if self.config["Interactive"] is False:
            raise ValueError("interactivity disabled in config")
        grid_options = {"editable": True}
        self.qgrid_widget = self.make_qgrid(
            self.database, show_toolbar=True, precision=3, grid_options=grid_options
        )
        return self.qgrid_widget

    def save_interactive(self):
        """
        saves changes made to the qgrid when interfactive edit mode has been
        used"""
        if self.config["Interactive"] is False:
            raise ValueError("interactivity disabled in config")
        self.database = self.qgrid_widget.get_changed_df()

    def get_database(self):
        """returns the pandas data frame"""
        return self.database

    def set_database(self, database):
        """set the pandas data frame"""
        self.database = database

    def save_to_file(self):
        """save the pandas dataframe to the root path of the catalogue
        file structure"""
        self.database.to_csv(
            os.path.join(self.base_path, self.file_name),
            index=False,
            index_label="Index",
        )

    def register_alias(self, row_id, alias):
        """create an alias for a material to easily access it from the
        catalogue"""
        index = None
        if isinstance(row_id, pd.core.series.Series):
            index = row_id.Index
        elif isinstance(row_id, int):
            index = row_id
        if index is None:
            raise ValueError(
                "row_id: {}".format(row_id)
                + " with type {}".format(type(row_id))
                + " not understood"
            )
        if alias in self.database.loc[:, "Alias"].values:
            if not self.database.at[index, "Alias"] == alias:
                raise ValueError(
                    "Alias {} ".format(alias)
                    + "already in use. Failed to "
                    + "add to catalogue."
                )

        self.database.at[index, "Alias"] = alias

    def get_material(self, identifier):
        """get a material from the catalogue using its alias or row number"""
        if isinstance(identifier, str):
            bool_array = self.database.Alias.values == identifier
            bool_list = bool_array.tolist()
            row_index = self.database.index[bool_list]
            if row_index.size == 0:
                raise ValueError(
                    "identifier {} does not ".format(identifier)
                    + "name a valid alias in the "
                    + "catalogue"
                )
            row = self.database.iloc[row_index[0], :]

        elif isinstance(identifier, int):
            row = self.database.iloc[identifier, :]
        else:
            raise ValueError("identifier must be of type str or int")
        file_path = os.path.normpath(
            os.path.join(self.base_path, row.Module, row.Path.replace("\\", "/"))
        )
        mat = Material(
            file_path=file_path, spectrum_type=row.SpectrumType, unit=row.Unit
        )
        return mat

    def make_reference_spectrum(
        self,
        config: Dict[
            str, Union[str, bool, Dict[str, bool], Dict[str, Union[float, str]]]
        ],
    ) -> None:
        """make the spectrum with which every material in the catalogue will
        be evaluated"""
        val = config["ReferenceSpectrum"]["Value"]
        spec_type = config["ReferenceSpectrum"]["SpectrumType"]
        unit = config["ReferenceSpectrum"]["Unit"]
        self.reference_spectrum = Spectrum(val, spectrum_type=spec_type, unit=unit)

    def read_ri_info_db(self, db_path):
        """read the file structure provided by the refractiveindex.info
        website"""
        self.rii_loader = {}
        self.rii_loader["db_path"] = db_path
        data = read_yaml_file(os.path.join(db_path, 'database', "library.yml"))
        dframe = pd.DataFrame(columns=Catalogue.META_DATA.keys())
        self.rii_loader["database_list"] = []
        self._iterate_shelves(data)

        dframe = pd.DataFrame(
            self.rii_loader["database_list"], columns=Catalogue.META_DATA.keys()
        )
        self.rii_loader = None
        return dframe

    def _iterate_shelves(self, shelves):
        """
        shelves is an ordered dict
        """
        for shelf in shelves:
            # shelf = shelf_data['SHELF']
            self.rii_loader["current_shelf"] = shelf["name"]
            books = shelf["content"]
            self._iterate_books(books)

    def _iterate_books(self, books):
        """
        iterate through the books on the shelf
        """
        for book in books:
            if "DIVIDER" in book.keys():
                self.rii_loader["current_divider"] = book["DIVIDER"]
                continue
            elif "BOOK" in book.keys():
                self.rii_loader["current_book"] = book["BOOK"]
                self.rii_loader["current_full_name"] = book["name"]
                pages = book["content"]
                self._iterate_pages(pages)

    def _iterate_pages(self, pages):
        """
        iterate the pages of the book and add to catalogue

        The pages of the refactiveindex.info catalogue are data files. We iterate
        over them and add the current shelf, book and page to the catalogue. A
        Material object is created from the data file.
        """
        for page in pages:
            if "DIVIDER" in page:
                continue
            elif "PAGE" in page:
                db_path = self.rii_loader["db_path"]
                rel_path = os.path.join("data", page["data"])
                full_file = os.path.join(db_path, rel_path)
                try:
                    mat = Material(
                        file_path=full_file,
                        spectrum_type="wavelength",
                        unit="micrometer",
                    )
                except OSError as exc:
                    warnings.warn(
                        "file {} ".format(full_file) + "could not be opened, skipping"
                    )
                    continue
                fullname = self.rii_loader["current_full_name"]
                content_dict = {
                    "Alias": "",
                    "Name": self.rii_loader["current_book"],
                    "FullName": fullname,
                    "Author": page["PAGE"],
                    "Path": os.path.normpath(rel_path),
                    "Module": "RefractiveIndexInfo",
                }
                content_dict["SpectrumType"] = "wavelength"
                content_dict["Unit"] = "micrometer"
                valid_range = mat.get_maximum_valid_range()
                content_dict["SpectrumLowerBound"] = valid_range[0]
                content_dict["SpectrumUpperBound"] = valid_range[1]
                content_dict["Reference"] = mat.meta_data["Reference"]
                content_dict["Comment"] = mat.meta_data["Comment"]
                try:
                    ref_spectrum = self.reference_spectrum
                    ref_index = mat.get_nk_data(ref_spectrum)
                except ValueError:
                    ref_index = np.nan + 1j * np.nan

                content_dict["N_Reference"] = np.real(ref_index)
                content_dict["K_Reference"] = np.imag(ref_index)
                self.rii_loader["database_list"].append(content_dict)

    def read_filmetrics_db(self, db_path):
        """read the file structure provided my filmetrics.com"""
        return self._read_text_db(db_path, "Filmetrics")

    def read_user_data_db(self, db_path):
        """read user data files"""
        return self._read_text_db(db_path, "UserData")

    def _read_text_db(self, db_path, database_name):
        """internal function used to load all txt, csv or yml
        files in a folder"""
        onlyfiles = [
            f for f in os.listdir(db_path) if os.path.isfile(os.path.join(db_path, f))
        ]

        dframe = pd.DataFrame(columns=Catalogue.META_DATA.keys())
        database_list = []
        for filename in onlyfiles:
            [name, ext] = os.path.splitext(filename)
            allowed_ext = {".txt", ".csv", ".yml"}
            if ext not in allowed_ext:
                continue

            mat = Material(
                file_path=os.path.join(db_path, filename),
                spectrum_type="wavelength",
                unit="nanometer",
            )

            content_dict = {}
            content_dict["Alias"] = ""
            content_dict["Name"] = name
            content_dict["FullName"] = mat.meta_data["FullName"]
            content_dict["Author"] = mat.meta_data["Author"]
            content_dict["Module"] = database_name
            content_dict["Comment"] = mat.meta_data["Comment"]
            content_dict["Reference"] = mat.meta_data["Reference"]
            content_dict["SpectrumType"] = "wavelength"
            content_dict["Unit"] = "nanometer"
            valid_range = mat.get_maximum_valid_range()
            content_dict["SpectrumLowerBound"] = valid_range[0]
            content_dict["SpectrumUpperBound"] = valid_range[1]
            content_dict["Path"] = os.path.normpath(filename)
            try:
                ref_index = mat.get_nk_data(self.reference_spectrum)
            except ValueError as valE:
                ref_index = float("nan") + 1j * float("nan")

            content_dict["N_Reference"] = np.real(ref_index)
            content_dict["K_Reference"] = np.imag(ref_index)
            database_list.append(content_dict)
        dframe = pd.DataFrame(database_list, columns=Catalogue.META_DATA.keys())
        return dframe


if __name__ == "__main__":
    rebuild_catalogue()
