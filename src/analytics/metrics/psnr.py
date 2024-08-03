import torch
from torch import nn
from skimage.metrics import peak_signal_noise_ratio
from .base import Metric


class PSNR(Metric):
    """Peak Signal-to-Noise Ratio implementation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, reconstructed_images, original_images):
        """
        Method of PSNR computing between
        reconstructed and original image.
        """
        iterator = zip(reconstructed_images, original_images)
        for reconstructed, original in iterator:
            original = original.permute(1, 2, 0)
            original = original.detach().cpu().numpy()

            reconstructed = reconstructed.permute(1, 2, 0)
            reconstructed = reconstructed.detach().cpu().numpy()

            psnr = peak_signal_noise_ratio(original, reconstructed)
            self._sum = torch.add(self._sum, psnr)
            self._total = torch.add(self._total, 1)
