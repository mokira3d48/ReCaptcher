import numpy as np
import torch
from .base import Metric


class CER(Metric):
    """Character Error Rate (CER) implementation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def levenshtein_distance(self, hyp, ref):
        # Initialiser une matrice de distance
        d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=int)

        # Remplir la première colonne et la première ligne
        for i in range(len(ref) + 1):
            d[i][0] = i

        for j in range(len(hyp) + 1):
            d[0][j] = j

        # Remplir la matrice de distance
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    cost = 0
                else:
                    cost = 1

                d[i][j] = min(d[i - 1][j] + 1,  # Suppression
                              d[i][j - 1] + 1,  # Insertion
                              d[i - 1][j - 1] + cost)  # Substitution

        return d[len(ref)][len(hyp)]

    def update_state(self, predicts, targets):
        """
        Method of CER computing between
        predicted sequence and targets sequences.
        """
        iterator = zip(predicts, targets)
        for predict, target in iterator:
            target = target.detach().cpu().numpy()
            predict = predict.detach().cpu().numpy()

            distance = self.levenshtein_distance(predict, target)
            cer = distance / len(target)
            self._sum = torch.add(self._sum, cer)
            self._total = torch.add(self._total, 1)
