import logging
import time

import torch
from .models.transformer import Transformer, Inference
from tokenizers import Tokenizer


LOG = logging.getLogger(__name__)


def load_from_file(model, file_path):
    """Function to load model from the model file path"""
    loaded = torch.load(file_path)
    model.load_state_dict(loaded)
    print("model is loaded from saving file path.")


def load_from_checkpoint(model, file_path):
    """Function to load model from checkpoint file"""
    loaded = torch.load(file_path)
    if 'model' in loaded:
        model.load_state_dict(loaded['model'])
        LOG.info("model is loaded from checkpoint file.")


class Prediction(object):
    """Prediction function"""

    def __init__(self, **kwargs):
        self.max_len = kwargs['max_len']
        self.model_config_fp = kwargs['model']['configFile']
        self.model_file_path = kwargs['model']['saved']
        self.ckp_file_path = kwargs['model']['checkpoint']

        self.src_vocab_fp = kwargs['vocab']['source']
        self.tgt_vocab_fp = kwargs['vocab']['target']

        src_tokenizer = Tokenizer.from_file(self.src_vocab_fp)
        tgt_tokenizer = Tokenizer.from_file(self.tgt_vocab_fp)

        self.src_vocab_size = len(src_tokenizer.get_vocab())
        self.tgt_vocab_size = len(tgt_tokenizer.get_vocab())

        model = Transformer.from_config_file(
            config_fp=self.model_config_fp,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
        )

        if self.model_file_path:
            load_from_file(model, self.model_file_path)
        elif self.ckp_file_path:
            load_from_checkpoint(model, self.ckp_file_path)

        self.inference = Inference(model=model,
                                   src_tokenizer=src_tokenizer,
                                   tgt_tokenizer=tgt_tokenizer,
                                   max_len=self.max_len)
    
    def run(self):
        """Method to run predinction loop with user interface"""
        while True:
            input_sentence = ''
            while not input_sentence:
                input_sentence = input("Type a french sentence > ")
                input_sentence = input_sentence.strip()

            output_sentence = self.inference([input_sentence])
            print("Sentence translated:")
            print("\033[92m")
            for c in output_sentence[0]:
                print(c, end='', flush=True)
                time.sleep(0.05)

            print("\033[0m")
            print()  # Skip on line;
