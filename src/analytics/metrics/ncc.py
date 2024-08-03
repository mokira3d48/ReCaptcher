import numpy as np
import torch
from .base import Metric


class NCC(Metric):
    """Peak Signal-to-Noise Ratio implementation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def ncc(self, image1, image2):
        numerator = np.sum(image1 * image2)
        denominator = np.linalg.norm(image1) * np.linalg.norm(image2)
        return numerator / denominator

    def update_state(self, reconstructed_images, original_images):
        """
        Method of NCC computing between
        reconstructed and original image.
        """
        iterator = zip(reconstructed_images, original_images)
        for reconstructed, original in iterator:
            original = original.permute(1, 2, 0)
            original = original.detach().cpu().numpy()

            reconstructed = reconstructed.permute(1, 2, 0)
            reconstructed = reconstructed.detach().cpu().numpy()

            ncc = self.ncc(original, reconstructed)
            self._sum = torch.add(self._sum, ncc)
            self._total = torch.add(self._total, 1)
