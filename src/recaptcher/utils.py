import os
import yaml


def get_file_paths_list(dir_path):
    """Returns the list of file paths contained on directory

    :param dir_path: The path to directory.
    :type dir_path: `str`
    :rtype: `list` of `str`
    """
    file_names = os.listdir(dir_path)
    file_paths = [os.path.join(dir_path, file_name)
                  for file_name in file_names]
    return file_paths


def load_args(file_path):
    """Fonction de chargement des donnees d'un fichier Yaml"""
    args = {}
    with open(file=file_path, mode='r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)

    return args
