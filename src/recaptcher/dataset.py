import gc
import os
import csv
import logging
import shutil

import numpy as np
import cv2 as cv
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from recaptcher.models.tokenizer import (CharacterTokenizer,
                                         Trainer as TokenizerTrainer)


LOG = logging.getLogger(__name__)


def preprocess_image(image, image_size=(224, 224)):
    """Function that is used to preprocess image"""
    image = cv.resize(image, dsize=image_size)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = np.asarray(image, dtype=np.float32)
    image = np.divide(image, 255.0)
    image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    image = np.transpose(image, (2, 1, 0))
    return image


def get_image_label(file_name):
    """Function which returns the image label by its file name"""
    file_name_split = file_name.split('_')
    file_name = file_name_split[1]
    file_name_split = file_name.split('.')
    file_name = file_name_split[0]
    return file_name


def preprocess_label(label, encode_fn):
    """Function which preprocess the label"""
    label_encoded = encode_fn(label)
    label_encoded = np.asarray(label_encoded, dtype=np.int64)
    return label_encoded


def label_padding(encoded_label, max_length, pad_token):
    """Function of sequence padding"""
    length = len(encoded_label)
    if length < max_length:
        padding = np.asarray([pad_token] * (max_length - length),
                             dtype=np.int64)
        # LOG.debug(padding.shape)
        # LOG.debug(encoded_label.shape)
        encoded_label = np.hstack([encoded_label, padding])
        return encoded_label

    if length > max_length:
        LOG.warning("Alert: An length is greater than the max length.")

    return encoded_label[:max_length]


class DatasetCollector:
    """Dataset collection"""

    def __init__(self, **kwargs):
        self.source_file = kwargs['source']['file']

        self.dataset_dir = kwargs['target']['dataset_dir']
        self.train_prob = kwargs['target']['train_prob']
        self.valid_prob = kwargs['target']['valid_prob']
        self.test_prob = kwargs['target']['test_prob']
        self.samples_count = kwargs['target']['samples_count']

        valid_test_sum = self.valid_prob + self.test_prob
        self.valid_prob = self.valid_prob / valid_test_sum

        # preparing of the tree path to the files;
        # "train.csv" "valid.csv", "test.csv";
        self.train_fp = os.path.join(self.dataset_dir, 'train/')
        self.valid_fp = os.path.join(self.dataset_dir, 'valid/')
        self.test_fp = os.path.join(self.dataset_dir, 'test/')

        for dir_path in [self.train_fp, self.valid_fp, self.test_fp]:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)

    def _get_random_ds_file_path(self):
        """Function of random selection of a dataset folder"""
        random_value = np.random.rand()
        if random_value <= self.train_prob:
            return self.train_fp
        else:
            random_value = np.random.rand()
            if random_value < self.valid_prob:
                return self.valid_fp
            else:
                return self.test_fp

    def run(self):
        """Execution function of running dataset collection"""
        if not os.path.isdir(self.source_file):
            LOG.warning("No such file at " + str(self.source_file) + ".")
            return

        file_names = os.listdir(self.source_file)
        tgt_dirs = {
            self.train_fp: len(self.train_fp),
            self.valid_fp: len(self.valid_fp),
            self.test_fp: len(self.test_fp),
        }
        for file_name in file_names:
            tgt_dir = self._get_random_ds_file_path()
            index = tgt_dirs[tgt_dir]

            src_fp = os.path.join(self.source_file, file_name)
            tgt_fp = os.path.join(tgt_dir, f"{index:08d}_{file_name}")
            shutil.copyfile(src_fp, tgt_fp)
            tgt_dirs[tgt_dir] += 1

        LOG.info(f"{tgt_dirs[self.train_fp]} train samples.")
        LOG.info(f"{tgt_dirs[self.valid_fp]} valid samples.")
        LOG.info(f"{tgt_dirs[self.test_fp]} test samples.")


class DatasetVocabBuilder:
    """Building of dataset vocab"""

    def __init__(self, **kwargs):
        self.dataset_dirs = kwargs['dataset_dirs']
        self.vocab_file = kwargs['vocab_file']
        self.pad_token = kwargs['pad_token']
        self.unk_token = kwargs['unk_token']

        # creation of parent directory of vocab file.
        vocab_file_dir_name = os.path.dirname(self.vocab_file)
        if not os.path.isdir(vocab_file_dir_name):
            os.makedirs(vocab_file_dir_name)

        self.tokenizer = CharacterTokenizer(pad_token=self.pad_token,
                                            unk_token=self.unk_token)

    def run(self):
        """Function of vocab building execution"""
        trainer = TokenizerTrainer(model=self.tokenizer)
        for dataset_dir in self.dataset_dirs:
            if not os.path.isdir(dataset_dir):
                LOG.warning(
                    "No such dataset directory at: " + str(dataset_dir)
                )
                continue

            file_names = os.listdir(dataset_dir)
            full_text = ''
            for file_name in file_names:
                label = get_image_label(file_name)
                full_text += label + ' '

            trainer.fit(text=full_text)

        trainer.save(self.vocab_file)


class DatasetBuilder:
    """Program to build the normalized dataset into NumPY file"""

    def __init__(self, **kwargs):
        self.mapping = kwargs['ds_mapping']
        self.vocab_file = kwargs['vocab_file']
        self.image_size = kwargs['image_size']
        self.max_seq_length = kwargs['max_seq_length']

        self.tokenizer = CharacterTokenizer()
        self.tokenizer.load_from_file(self.vocab_file)
        LOG.debug(str(self.tokenizer.vocab))

    def run(self):
        encode_fn = self.tokenizer.encode
        pad_token = self.tokenizer.pad
        for src_dir, tgt_dir in self.mapping.items():
            img_dir = os.path.join(tgt_dir, "img")
            lab_dir = os.path.join(tgt_dir, "lab")
            len_dir = os.path.join(tgt_dir, "len")
            dirs = [img_dir, lab_dir, len_dir]
            for dir_path in dirs:
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)

            file_names = os.listdir(src_dir)
            for src_file_name in tqdm(file_names):
                src_file_path = os.path.join(src_dir, src_file_name)
                try:
                    image = cv.imread(src_file_path)
                    image_preprocessed = preprocess_image(image,
                                                          self.image_size)
                except cv.error as e:
                    LOG.warning(str(e))
                    continue

                label = get_image_label(src_file_name)
                length = np.asarray(len(label), dtype=np.int64)
                label_preprocessed = preprocess_label(label, encode_fn)
                # LOG.debug(label)
                # LOG.debug(label_preprocessed)

                label_padded = label_padding(label_preprocessed,
                                             max_length=self.max_seq_length,
                                             pad_token=pad_token)

                img_fp = os.path.join(img_dir, f"{src_file_name}.npy")
                lab_fp = os.path.join(lab_dir, f"{src_file_name}.npy")
                len_fp = os.path.join(len_dir, f"{src_file_name}.npy")
                np.save(file=img_fp, arr=image_preprocessed)
                np.save(file=lab_fp, arr=label_padded)
                np.save(file=len_fp, arr=length)


class CaptchaDataset(Dataset):

    def __init__(self, dataset_fp):
        super().__init__()
        self.dataset_fp = dataset_fp

        self.img_dir = os.path.join(dataset_fp, "img")
        self.lab_dir = os.path.join(dataset_fp, "lab")
        self.len_dir = os.path.join(dataset_fp, "len")

        self.img_file_names = os.listdir(self.img_dir)
        self.lab_file_names = os.listdir(self.lab_dir)
        self.len_file_names = os.listdir(self.len_dir)

        assert len(self.img_file_names) == len(self.lab_file_names) \
               and len(self.lab_file_names) == len(self.len_file_names), (
            "The number of image is not equal to number of label or length."
        )

    def __len__(self):
        return len(self.img_file_names)

    def remove_sample(self, index):
        del self.img_file_names[index]
        del self.lab_file_names[index]
        del self.len_file_names[index]
        gc.collect()

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range: " + str(index))

        file_name = self.img_file_names[index]
        img_file_path = os.path.join(self.img_dir, file_name)
        if not os.path.isfile(img_file_path):
            self.remove_sample(index)
            return self.__getitem__(index)

        lab_file_path = os.path.join(self.lab_dir, file_name)
        len_file_path = os.path.join(self.len_dir, file_name)
        if not os.path.isfile(lab_file_path) \
                and not os.path.isfile(len_file_path):
            self.remove_sample(index)
            return self.__getitem__(index)

        image = np.load(img_file_path)
        label = np.load(lab_file_path)
        length = np.load(len_file_path)

        image = torch.tensor(image)
        label = torch.tensor(label, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)

        return image, label, length


def main():
    from torch.utils.data import DataLoader

    ds = CaptchaDataset('outputs/dataset_norm/train')
    loader = DataLoader(ds, batch_size=2, num_workers=4)
    for x, y, l in loader:
        print(x.shape, y, l)
        input()


if __name__ == '__main__':
    main()
