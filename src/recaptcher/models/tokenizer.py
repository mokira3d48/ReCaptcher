import os
import logging
import json


LOG = logging.getLogger(__name__)


class CharacterTokenizer(object):
    """Character tokenization"""

    def __init__(self, pad_token=None, unk_token=None):
        self.vocab = []
        self.pad_token = pad_token
        self.unk_token = unk_token

    @property
    def vocab_size(self):
        """int: the vocab size"""
        adding = 0
        if self.unk_token:
            adding += 1

        if self.pad_token:
            adding += 1

        return len(self.vocab) + adding

    def load_from_file(self, file_path):
        """Method to load characters vocab from json file"""
        if not os.path.isfile(vocab_file_path):
            raise FileNotFoundError(
                f"No such vocab file at {vocab_file_path}"
            )

        json_model = {}
        with open(vocab_file_path) as file:
            json_model = json.load(vocab_file_path)

        self.vocab = json_model.get('vocab', [])
        if '<pad>' in self.vocab:
            self.pad_token = self.vocab.index('<pad>')

        if '<unk>' in self.vocab:
            self.unk_token = self.vocab.index('<unk>')

    def encode(self, strings):
        """Function of string encoding"""
        if not hasattr(strings, '__iter__'):
            strings = [strings]

        output_sequences = []
        for string in strings:
            encoded_string = []
            for character in string:
                if character in self.main_chars:
                    index = self.main_chars.index(character)
                    encoded_string.append(index)
                else:
                    msg = "Character " + str(character) + " is unknown."
                    LOG.warning(msg)
                    if not self.unk_token:
                        raise IndexError(msg)

                    encoded_string.append(self.unk_token)

            output_sequences.append(encoded_string)

        return output_sequences

    def decode(self, encoded_strings):
        """Method of sequence decoding"""
        if not hasattr(encoded_strings, '__iter__'):
            encoded_strings = [encoded_strings]

        output_strings = []
        final_index = self.vocab_size - 1
        for encoded_string in encoded_strings:
            decoded_string = ''
            for code in encoded_string:
                if code < 0 or code > final_index:
                    raise ValueError("The code " + str(code) + " is invalid.")

                character = self.vocab[code]
                if character == '<pad>':
                    continue

                decoded_string += character

            output_strings.append(decoded_string)

        return output_strings


class CharactersTokenizerTrainer(object):
    """Represent algorithm of training of a character tokenizer model"""

    def __init__(self, model, text=None, file_path=None):
        self.model = model
        self.text = text
        self.file_path = file_path

    def fit(self):
        """Function which fit the model of character tokenizer"""
        string_text = ''
        if self.text:
            string_text = self.text

        if not string_text:
            if self.file_path and os.path.isfile(self.file_path):
                with open(string_text, mode='r', encoding='UTF-8') as file:
                    string_text = file.read()

        if not string_text:
            LOG.warning("No text or content file text given.")
            return

        pad = self.model.pad_token
        unk = self.model.unk_token
        if pad:
            self.model.vocab.append('<pad>')

        if unk:
            self.model.vocab.append('<unk>')

        for character in string_text:
            if character in self.model.vocab:
                continue

            self.model.vocab.append(character)

