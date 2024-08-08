import os
import csv
import logging
import shutil

import numpy as np
import torch
# from tqdm import tqdm
from torch.utils.data import Dataset
from .models.tokenizer import CharacterTokenizer, Trainer as TokenizerTrainer


LOG = logging.getLogger(__name__)


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
                file_name_split = file_name.split('_')
                file_name = file_name_split[1]
                file_name_split = file_name.split('.')
                file_name = file_name_split[0]
                full_text += file_name + ' '

            trainer.fit(text=full_text)

        trainer.save(self.vocab_file)


class BilingualDataset(Dataset):

    def __init__(
            self,
            dataset_fp,
            src_column_name,
            tgt_column_name,
            src_tokenizer,
            tgt_tokenizer,
            src_seq_max_len=100,
            tgt_seq_max_len=100,
    ):
        super().__init__()
        self.dataset_fp = dataset_fp
        self.src_column_name = src_column_name
        self.tgt_column_name = tgt_column_name
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_seq_max_len = src_seq_max_len
        self.tgt_seq_max_len = tgt_seq_max_len

        self.sent_count = self._get_rows_count(src_column_name, dataset_fp)

        self.src_pad = self.src_tokenizer.encode('<pad>').ids[0]
        self.src_sos = self.src_tokenizer.encode('<sos>').ids[0]
        self.src_eos = self.src_tokenizer.encode('<eos>').ids[0]

        self.tgt_pad = self.tgt_tokenizer.encode('<pad>').ids[0]
        self.tgt_sos = self.tgt_tokenizer.encode('<sos>').ids[0]
        self.tgt_eos = self.tgt_tokenizer.encode('<eos>').ids[0]

        self.src_special_tokens = [self.src_pad,
                                   self.src_sos,
                                   self.src_eos,
                                   self.src_seq_max_len]
        self.tgt_special_tokens = [self.tgt_pad,
                                   self.tgt_sos,
                                   self.tgt_eos,
                                   self.tgt_seq_max_len]

        self.src_iter = self._get_rows_iterator(src_column_name, dataset_fp)
        self.tgt_iter = self._get_rows_iterator(tgt_column_name, dataset_fp)

    @staticmethod
    def padding(sequence, pad_id, sos_id, eos_id, max_len):
        sequence = list(sequence)
        sequence.insert(0, sos_id)
        sequence.append(eos_id)

        seq_len = len(sequence)
        if seq_len < max_len:
            padded_seq = sequence + ([pad_id] * (max_len - seq_len))
            return padded_seq

        return sequence[:max_len]

    def get_next(self):
        try:
            next_src_sentence = next(self.src_iter)
            next_tgt_sentence = next(self.tgt_iter)
            return next_src_sentence, next_tgt_sentence
        except StopIteration:
            self.src_iter = self._get_rows_iterator(self.src_column_name,
                                                    self.dataset_fp)
            self.tgt_iter = self._get_rows_iterator(self.tgt_column_name,
                                                    self.dataset_fp)
            return self.get_next()

    def __len__(self):
        return self.sent_count

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range: " + str(index))

        src_sentence, tgt_sentence = self.get_next()

        src_encoded = self.src_tokenizer.encode(src_sentence)
        tgt_encoded = self.tgt_tokenizer.encode(tgt_sentence)
        src_token_ids = self.padding(src_encoded.ids, *self.src_special_tokens)
        tgt_token_ids = self.padding(tgt_encoded.ids, *self.tgt_special_tokens)

        src_token_ids = torch.tensor(src_token_ids, dtype=torch.long)
        tgt_token_ids = torch.tensor(tgt_token_ids, dtype=torch.long)
        return src_token_ids, tgt_token_ids


def main():
    from torch.utils.data import DataLoader

    ds = BilingualDataset('outputs/datasets/train.csv',
                          'french',
                          'english',
                          'outputs/vocab/french_vocab.json',
                          'outputs/vocab/english_vocab.json')
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    for x, y, in loader:
        print(x, y)
        input()


if __name__ == '__main__':
    main()
