import random
import logging
from unittest import TestCase
from recaptcher.models.tokenizer import CharactersTokenizer


LOG = logging.getLogger(__name__)


def add_zeros_randomly(numbers, num_zeros):
    """
    Add zeros at random positions in an integers list.

    Args:
        numbers (list): List of integers.
        num_zeros (int): Number of zeros to add.

    Returns:
        list: The modified list with 0 added at random positions.
    """
    indices = random.sample(range(len(numbers)), num_zeros)
    for i in sorted(indices, reverse=True):
        numbers.insert(i, 0)

    return numbers


class CharacterTokenizerTest(TestCase):
    """Characters tokenization test cases"""

    def setUp(self):
        self.text1 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.text2 = "abcdefghijklmnopqrstuvwxyz"
        self.text3 = "0123456789"

        self.sample_vf = ''
        self.sample_vf_padding = ''
        self.sample_vf_unknown = ''

    def test_normal_case(self):
        result1 = list(range(26))
        result2 = list(range(26, 52))
        result3 = list(range(52, 62))

        tokenizer = CharactersTokenizer.load_from_file(self.sample_vf)
        inputs = [self.text1, self.text2, self.text3]
        outputs = [result1, result2, result3]
        for inp, out in zip(inputs, outputs):
            text_tokenized = tokenizer.encode(inp)
            LOG.debug("input: " + str(inp) + " output: " + str(text_tokenized))
            self.assertEqual(text_tokenized, out)

            decoded_text = tokenizer.decode(out)
            LOG.debug("output: " + str(out) + " input: " + str(decoded_text))
            self.assertEqual(decoded_text, out)

    def test_with_padding(self):
        """
        We test the tokenizer with padding characters.
        """
        result1 = list(range(26))
        result2 = list(range(26, 52))
        result3 = list(range(52, 62))
        tokenizer = CharactersTokenizer.load_from_file(self.sample_vf_padding)
        pad_token = tokenizer.pad_token  # we use 1000 as index;
        result1 = ([pad_token]*12) + result1
        result2 = result2 + ([pad_token]*54)
        result3 = ([pad_token]*39) + result3 + ([pad_token]*23)

        inputs = [self.text1, self.text2, self.text3]
        outputs = [result1, result2, result3]
        for inp, out in zip(inputs, outputs):
            decoded_text = tokenizer.decode(out)
            LOG.debug("output: " + str(out) + " input: " + str(decoded_text))
            self.assertEqual(decoded_text, out)

    def test_with_unknown_token(self):
        """
        We test the tokenizer on a string that contains
        the unknown characters.
        """
        ...

    def test_decoding(self):
        """
        We test the decoding function without padding char
        and unknown char.
        """
        ...

    def test_fit(self):
        """
        We test fit function.
        """
