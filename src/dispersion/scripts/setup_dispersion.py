#!/usr/bin/env python
"""
setup the disperion database file structure and configuration file
"""
import os
import tempfile
import numpy as np
from dispersion import Material, Writer, Interpolation, Catalogue
from dispersion.config import default_config, write_config

def get_root_dir(conf):
    """
    get the root dir from the user
    """
    question = ("path for root directory of the catalogue file" +
                " system [default: {}]> ")
    default = conf['Path']
    validator = os.path.isabs
    data_name = "root directory"
    return ask_and_confirm(question, default, validator, data_name)

def get_catalogue_name(conf):
    """
    get the catalogue file name from the user
    """
    question = ("name of the catalogue file" +
                " [default: {}]> ")
    default = conf['File']
    validator = valid_file_name
    data_name = "catalogue file name"
    return ask_and_confirm(question, default, validator, data_name)

def ask_and_confirm(question, default, validator, data_name, confirm=True):
    """
    Returns
    -------
    user_input: str
        the data from the user
    confirmed_input: bool
        true if the input was confirmed by the user

    Parameters
    ----------
    question: str
        the question to prompt the user input
    default: str
        the default value of this value
    validator: function
        function to validate the input
    data_name: str
        name of the data that is being input
    """
    user_input = ask(question, default, validator)
    confirmation_question = (f"confirm {data_name} as " + f"{user_input}? [y/n]> ")
    return [user_input, get_confirmation(confirmation_question)]


def ask(question, default, validator):
    """
    ask for user input with default value and then validate
    """
    valid_input = False
    while not valid_input:
        user_input = input(question.format(default))
        if user_input == "":
            user_input = default
        if validator(user_input):
            valid_input = True
        else:
            print("input is not valid")
    return user_input

def get_confirmation(question):
    """
    get a yes/no answer to a question
    """
    confirmed_input = False
    while not confirmed_input:
        confirmation1 = input(question)
        if confirmation1 in {'y', 'yes'}:
            confirmed_input = True
        elif confirmation1 in {'n', 'no'}:
            confirmed_input = False
            break
        else:
            print("input invalid")
    return confirmed_input

def valid_file_name(filename):
    """
    test if filename is valid

    create a file with the filename in a temporary directory and delete the
    directory afterwards.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, filename)
        try:
            open(file_path, 'r')
            return True
        except IOError:
            try:
                open(file_path, 'w')
                return True
            except IOError:
                return False

def install_modules(conf):
    """
    make a subfolder for each module and ask to download files
    """

    install_funcs = {"UserData":install_userdata,
                     "RefractiveIndexInfo":install_rii}

    for module in conf['Modules']:
        if module == "UserData":
            install = True
        else:
            question = f"install module {module}? [y/n]> "
            install = get_confirmation(question)

        conf['Modules'][module] = install
        if install:
            module_dir = os.path.join(conf['Path'], module)
            if not os.path.isdir(module_dir):
                os.mkdir(module_dir)
            install_funcs[module](module_dir, conf)
    return conf

def install_userdata(module_dir, conf):
    make_example_txt(module_dir)
    make_example_yaml(module_dir)

def make_example_txt(dir_path):
    test_data = np.array([[400., 1.7, 0.1],
                          [500., 1.6, 0.05],
                          [600., 1.5, 0.0],
                          [700., 1.4, 0.0]])
    mat = Material(tabulated_nk=test_data,
                   spectrum_type='wavelength', unit='nanometer')
    mat.meta_data['Reference'] = "Literature reference to the data"
    mat.meta_data['Comment'] = "Any additional information goes here"
    mat.meta_data['Name'] = "Short name of the material"
    mat.meta_data['FullName'] = "Full name of the material"
    mat.meta_data['Author'] = "The author of this data file"
    mat.meta_data['MetaComment'] = " This is a multiline meta-comment\n" + \
                                   " which provides information not\n" + \
                                   " in metadata"
    filepath = os.path.join(dir_path, "example_file.txt")
    write = Writer(filepath, mat)
    write.write_file()

def make_example_yaml(dir_path):
    model_params = {'name': 'Sellmeier',
                    'specrtrum_type':'wavelength',
                    'unit':'micrometer',
                    'valid_range':np.array([0.350, 2.0]),
                    'parameters': np.array([0, 1.0, 0.05,
                                            2.0, 0.1,
                                            10., 25.])}
    mat = Material(model_kw=model_params, spectrum_type='wavelength', unit='micrometer')
    mat.meta_data['Reference'] = "Literature reference to the data"
    mat.meta_data['Comment'] = "Any additional information goes here"
    mat.meta_data['Name'] = "Short name of the material"
    mat.meta_data['FullName'] = "Full name of the material"
    mat.meta_data['Author'] = "The author of this data file"
    mat.meta_data['MetaComment'] = " This is a multiline meta-comment\n" + \
                                   " which provides information not\n" + \
                                   " in metadata"
    k_data = np.array([[400., 0.1],
                       [500., 0.05],
                       [600., 0.0],
                       [700., 0.0]])
    interp = Interpolation(k_data, unit='nm')
    mat.data['imag'] = interp
    filepath = os.path.join(dir_path, "example_file2.yml")
    write = Writer(filepath, mat)
    write.write_file()

def install_rii(module_dir, conf):
    """
    download the refractive index info database from github
    """
    question = ("download the refractive index info database from github?" +
                " (required python package <GitPython>)" +
                " [y/n]> ")
    if install := get_confirmation(question):
        from git import Repo
        git_url = "https://github.com/polyanskiy/refractiveindex.info-database.git"
        #install_dir = os.path.join(conf['Path'], "RefractiveIndexInfo")
        Repo.clone_from(git_url, module_dir)

def maybe_rebuild_catalogue(conf):
    question = "rebuild catalogue? [y/n]> "
    if rebuild := get_confirmation(question):
        cat = Catalogue(config=conf, rebuild= 'All')
        cat.save_to_file()

def main():
    conf = default_config()
    print("This script will provide a default configuration for the \n"+
          "dispersion package")
    confirmed_valid_path = False
    while not confirmed_valid_path:
        [path, confirmed_valid_path] = get_root_dir(conf)
    conf['Path'] = path
    #print("Path will be se to: {}".format(path))
    confirmed_db_nane = False
    while not confirmed_db_nane:
        [name, confirmed_db_nane] = get_catalogue_name(conf)
    conf['File'] = name

    #print("Filename will be set to {}".format(name))
    conf = install_modules(conf)
    write_config(conf)

    maybe_rebuild_catalogue(conf)





if __name__ == "__main__":
    main()
