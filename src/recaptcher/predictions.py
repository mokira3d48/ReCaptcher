import logging
import time

import torch
# from tokenizers import Tokenizer

import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk  # noqa

import cv2 as cv

from recaptcher.models.preprocessing import (CharacterTokenizer,
                                             ImagePreprocessingFunction)
from recaptcher.models.resnet_gru import (Encoder, Decoder, CRNN,
                                          InputFunction, OutputFunction)


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


class Prediction:
    """Prediction function"""

    def __init__(self, **kwargs):
        self.image_size = kwargs['image_size']
        self.encoder_config_fp = kwargs['encoder']['configFile']
        self.model_file_path = kwargs['model']['saved']
        self.ckp_file_path = kwargs['model']['checkpoint']

        self.vocab_fp = kwargs['vocab']['file']

        tokenizer = CharacterTokenizer()
        tokenizer.load_from_file(self.vocab_fp)

        encoder = Encoder.from_config_file(self.encoder_config_fp)
        decoder = Decoder(num_chars=tokenizer.vocab_size)
        self.model = CRNN(encoder, decoder)

        if self.model_file_path:
            load_from_file(self.model, self.model_file_path)
        elif self.ckp_file_path:
            load_from_checkpoint(self.model, self.ckp_file_path)

        image_preprocessing = ImagePreprocessingFunction(self.image_size)
        self.input_fn = InputFunction(image_preprocessing)
        self.output_fn = OutputFunction(token_decoding=tokenizer.decode,
                                        blank_index=tokenizer.pad_index)

    def perform(self, file_path):
        image = None
        try:
            image = cv.imread(file_path)
        except cv.Error as e:
            LOG.error(str(e))
            return ''

        processed_images = self.input_fn([image])
        predictions = self.model.forward(processed_images)
        strings = self.output_fn(predictions)
        return strings[0]


class Main:
    def __init__(self, **kwargs):
        self.prediction = Prediction(**kwargs)
        self.image_size = kwargs['image_size']
        self.master = tk.Tk()
        self.master.title("Captcha prediction")

        self.canvas = tk.Canvas(self.master, width=500, height=500)
        self.canvas.pack()

        self.label = tk.Label(self.master, text="")
        self.label.pack(pady=10)

        self.load_button = tk.Button(self.master,
                                     text="Load an image",
                                     command=self.load_image)
        self.load_button.pack(pady=10)

    def load_image(self):
        # Ouvrir une boîte de dialogue pour sélectionner un fichier
        file_path = tkinter.filedialog.askopenfilename(
            title="Choose an image file",
            filetypes=(("PNG Image files", "*.png"),
                       ("JPEG Image files", "*.jpeg"),
                       ("JPG Image files", "*.jpg"),
                       ("All files", "*.*"))
        )

        if file_path:
            image = Image.open(file_path)

            image.thumbnail(self.image_size, Image.HUFFMAN_ONLY)
            photo_image = ImageTk.PhotoImage(image)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
            self.canvas.image = photo_image

            string_predicted = self.prediction.perform(file_path)
            self.label.config(text=string_predicted)

    def run(self):
        self.master.mainloop()
