import os
import json
import logging
import glob
from datetime import datetime
import torch
from torch import nn


LOG = logging.getLogger(__name__)


# def get_variables(string):
# 	"""Funtion to get variable from string formatting
#
# 	:param string: The initial string.
# 	:type string: str
# 	:rtype: `list` of `str`
# 	"""
# 	pattern = r"{([^}]*)}"
# 	return re.findall(pattern, string)


def get_file_name(self, file_name):
    """Funtion to get the file name without its extension

    :param file_name: The file name with its extension.
    :type file_name: str
    :rtype: str
    """
    fn_split = file_name.split('.')
    file_name_w_ext = fn_split[0]
    return file_name_w_ext


class CheckpointManager(object):

    META_FILE_NAME = 'metadata.json'
    DEFAULT_FN_FORMAT = '_checkpoint_{ckpid}.pth'

    def __init__(
        self,
        name,
        ckp_dir,
        fn_format = None,
        meta_fn = META_FILE_NAME,
        last_ckp_max_count = 5,
    ):
        self._name = name
        self._ckp_dir = ckp_dir
        self._fn_format = (fn_format if fn_format \
            else "{name}_{date}_{time}") + self.DEFAULT_FN_FORMAT
        self._meta_fn = meta_fn
        self._last_ckp_max_count = last_ckp_max_count

        if not os.path.isdir(ckp_dir):
            os.makedirs(ckp_dir)

        self._state = {}
        self._latest_fn = ''
        self._id = 0
        self._lasted_checkpoints = self._get_checkpoint_files(ckp_dir)
        self.remove_old_checkpoints()

    @property
    def latest_fn(self):
        """:str: The latest file name"""
        return self._latest_fn

    @property
    def latest_id(self):
        """:int: The latest id"""
        return self._id

    def _get_checkpoint_files(self, dir_path):
        """
        Récupère la liste des fichiers dans un répertoire
        dont le nom suit un certain pattern.

        Args:
            directory (str): Le chemin du répertoire à explorer.
            pattern (str): Le pattern à rechercher dans les noms de fichiers.

        Returns:
            list: La liste des chemins complets des fichiers correspondants.
        """
        # Construire le pattern de recherche
        search_pattern = os.path.join(dir_path, "*_checkpoint_*.pth")
        # Utiliser glob pour récupérer la liste des fichiers
        file_paths = glob.glob(search_pattern)
        file_paths = sorted(file_paths)
        return file_paths

    def remove_old_checkpoints(self):
        """Methode de suppression des vieux points de suavegardes"""
        if len(self._lasted_checkpoints) <= self._last_ckp_max_count:
            return
        
        old_ckp_files = self._lasted_checkpoints[:-self._last_ckp_max_count]
        for file_path in old_ckp_files:
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        start = self._last_ckp_max_count
        self._lasted_checkpoints = self._lasted_checkpoints[-start:]
        self._lasted_checkpoints = sorted(self._lasted_checkpoints)

    def put(self, attr_name, value):
        self._state[attr_name] = value

    def get(self, attr_name):
        return self._state.get(attr_name)

    def save(self, latest_id = None):
        """Saving method of a checkpoint

        :param latest_id: The latest id will be used to save
                          this checkpoint into hard disk.
        :type latest_id: int
        """
        if latest_id and int(latest_id) == 0:
            raise ValueError("Value `0` is not allowed.")

        data_saved = {}
        for attr, value in self._state.items():
            if not isinstance(value, nn.Module):
                data_saved[attr] = value
            else:
                weights = value.state_dict()
                data_saved[attr] = weights

        current_dt = datetime.now()
        time = current_dt.strftime('%H%M%S')
        date = current_dt.strftime('%Y%m%d')
        self._id = (self._id+1) if not latest_id else int(latest_id)
        ckp_fn = self._fn_format.format(date=date,
                                        time=time,
                                        name=self._name,
                                        ckpid=self._id)
        ckp_fp = os.path.join(self._ckp_dir, ckp_fn)

        try:
            torch.save(data_saved, ckp_fp)
        except RuntimeError as e:
            LOG.warning(str(e))

        metafp = os.path.join(self._ckp_dir, self._meta_fn)
        with open(metafp, mode='w', encoding='UTF-8') as metaf:
            metadata = {'latest_fn': ckp_fn, 'id': self._id}
            json_format = json.dumps(metadata, indent=4)
            metaf.write(json_format)

        self._lasted_checkpoints.append(ckp_fp)
        self.remove_old_checkpoints()

    def _retrieve_latest(self):
        if self._latest_fn:
            return

        metafp = os.path.join(self._ckp_dir, self._meta_fn)
        if not os.path.isfile(metafp):
            return

        with open(metafp, mode='r', encoding='UTF-8') as metaf:
            latest = json.loads(metaf.read())
            self._latest_fn = latest.get('latest_fn')
            self._id = int(latest.get("id", 0))

    def is_available(self):
        self._retrieve_latest()
        if not self._latest_fn:
            return False

        latest_fp = os.path.join(self._ckp_dir, self._latest_fn)
        return os.path.isfile(latest_fp)

    def load_latest(self):
        """Method to load latest checkpoint file from directory checkpoint"""
        self._retrieve_latest()
        if not self._latest_fn:
            return

        file_path = os.path.join(self._ckp_dir, self._latest_fn)
        if not os.path.isfile(file_path):
            return

        if self._id == 0:
            # the id is not found ion the metafile.
            # We can recovery it from latest file name.
            # get name without ext;
            file_name_w = get_file_name(self._latest_fn)
            fn_split = file_name_w.split('_')
            self._id = int(fn_split[-1])

        data_loaded = torch.load(file_path)
        state_loaded = {}
        for attr_name, value in self._state.items():
            if attr_name not in data_loaded:
                continue

            if isinstance(value, nn.Module):
                value.load_state_dict(data_loaded[attr_name])
                state_loaded[attr_name] = value
            else:
                state_loaded[attr_name] = data_loaded[attr_name]

        for attr, value in state_loaded.items():
            self._state[attr] = value
