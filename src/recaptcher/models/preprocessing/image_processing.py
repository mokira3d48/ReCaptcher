import logging

import numpy as np
import cv2 as cv


LOG = logging.getLogger(__name__)


class ImagePreprocessingFunction:
    """
    An implementation of image preprocessing
    which will be used to preprocess image before
    to make prediction with model.
    """

    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def preprocess(self, image):
        image = cv.resize(image, dsize=self.image_size)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = np.asarray(image, dtype=np.float32)
        image = np.divide(image, 255.0)
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        image = np.transpose(image, (2, 1, 0))
        LOG.debug("image shape: " + str(image.shape))
        return image

    def __call__(self, image):
        return self.preprocess(image)
