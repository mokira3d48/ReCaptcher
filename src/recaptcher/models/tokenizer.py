import os
import logging
import json


LOG = logging.getLogger(__name__)


class CharacterTokenizer(object):
    """Character tokenization"""

    def __init__(self, pad_token=False, unk_token=False):
        self.vocab = []
        self.pad_token = pad_token
        self.unk_token = unk_token

    @property
    def vocab_size(self):
        """int: the vocab size"""
        return len(self.vocab)

    @property
    def pad(self):
        """int: Padding ID"""
        if '[pad]' in self.vocab:
            return self.vocab.index('[pad]')

    @property
    def unk(self):
        """int: Padding ID"""
        if '[unk]' in self.vocab:
            return self.vocab.index('[unk]')

    def load_from_file(self, file_path):
        """Method to load characters vocab from json file"""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"No such vocab file at {file_path}"
            )

        json_model = {}
        with open(file_path, mode='r', encoding='UTF-8') as file:
            json_model = json.load(file)

        self.vocab = json_model.get('vocab', [])
        self.pad_token = int('[pad]' in self.vocab)
        self.unk_token = int('[unk]' in self.vocab)

    def encode(self, strings):
        """Function of string encoding"""
        if not hasattr(strings, '__iter__'):
            strings = [strings]

        output_sequences = []
        for string in strings:
            encoded_string = []
            for character in string:
                if character in self.vocab:
                    index = self.vocab.index(character)
                    encoded_string.append(index)
                else:
                    msg = "Character " + str(character) + " is unknown."
                    LOG.warning(msg)
                    if not self.unk_token:
                        raise IndexError(msg)

                    encoded_string.append(self.vocab.index('[unk]'))

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
                if character == '[pad]':
                    continue

                decoded_string += character

            output_strings.append(decoded_string)

        return output_strings


class Trainer(object):
    """Represent algorithm of training of a character tokenizer model"""

    def __init__(self, model):
        self.model = model

    def fit(self, text=None, file_path=None):
        """Function which fit the model of character tokenizer"""
        string_text = ''
        if text:
            string_text = text

        if not string_text:
            if file_path and os.path.isfile(file_path):
                with open(string_text, mode='r', encoding='UTF-8') as file:
                    string_text = file.read()

        if not string_text:
            LOG.warning("No text or content file text given.")
            return

        pad = self.model.pad_token
        unk = self.model.unk_token
        if pad and '[pad]' not in self.model.vocab:
            self.model.vocab.append('[pad]')

        if unk and '[unk]' not in self.model.vocab:
            self.model.vocab.append('[unk]')

        for character in string_text:
            if character in self.model.vocab:
                continue

            self.model.vocab.append(character)

        self.model.vocab.sort()

    def save(self, model_file_path):
        """Method of model saving into json file"""
        with open(model_file_path, mode='w', encoding='UTF-8') as file:
            stringify = json.dumps({'vocab': self.model.vocab})
            file.write(stringify)
