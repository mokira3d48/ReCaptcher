import json


class CharacterTokenizer(object):
    """Character tokenization"""

    def __init__(self, pad_token=None, unk_token=None):
        self.main_chars = []
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

        return len(self.main_chars) + adding

    def from_file(self, file_path):
        """Method to load characters vocab from json file"""
        if not os.path.isfile(vocab_file_path):
            raise FileNotFoundError(
                f"No such vocab file at {vocab_file_path}"
            )

        json_model = {}
        with open(vocab_file_path) as file:
            json_model = json.load(vocab_file_path)

        special_chars = json_model.get('spacial_chars', {})
        self.pad_token = special_chars.get('pad')
        self.unk_token = special_chars.get('unk')
        self.main_chars = json_model.get('main_chars', {})

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
                    encoded_string.append(self.unk_token)

            output_sequences.append(encoded_string)

        return output_sequences

    def decode(self, encoded_strings):
        """Method of sequence decoding"""
        ...
