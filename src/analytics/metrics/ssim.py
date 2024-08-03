import torch
from torch import nn
from skimage.metrics import structural_similarity
from .base import Metric


class SSIM(Metric):
    """Peak Signal-to-Noise Ratio implementation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, reconstructed_images, original_images):
        """
        Method of SSIM computing between
        reconstructed and original image.
        """
        iterator = zip(reconstructed_images, original_images)
        for reconstructed, original in iterator:
            original = original.permute(1, 2, 0)
            original = original.detach().cpu().numpy()

            reconstructed = reconstructed.permute(1, 2, 0)
            reconstructed = reconstructed.detach().cpu().numpy()

            ssim = structural_similarity(original,
                                         reconstructed,
                                         channel_axis=-1,
                                         data_range=1)
            self._sum = torch.add(self._sum, ssim)
            self._total = torch.add(self._total, 1)
