from typing import Union, Literal

import numpy as np
import torch

from global_variable import EPSILON
from pretty_print import pretty_print


class Similarity:

    @staticmethod
    def pearson(
            input: Union[torch.Tensor, np.ndarray],
            return_type: Literal["torch", "numpy"] = "numpy"
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        :param input: (sample, channel, pixel)
        :param return_type: (torch, numpy)
        :return: pearson_correlation: (channel, channel)
        """
        pretty_print("calculating pearson correlation... (network.util 30)")
        if isinstance(input, torch.Tensor):
            input = input.detach().cpu().numpy()
        _, channel, _ = input.shape
        mean_input = np.mean(input, axis=0)  # (channel, pixel)
        center_input: np.ndarray = input - mean_input  # (sample, channel, pixel)
        cov = np.mean(
            np.matmul(
                center_input,  # (sample, channel, pixel)
                center_input.transpose(0, 2, 1)  # (sample, pixel, channel)
            ),  # (sample, channel, channel)
            axis=0
        )  # (channel, channel)
        diag = np.diag(cov).reshape(channel, -1)  # (channel, 1)
        var = np.matmul(diag, diag.T)  # (channel, channel)
        pearson_correlation: np.ndarray = cov / (np.sqrt(var) + EPSILON)  # (channel, channel)
        if return_type == "numpy":
            return pearson_correlation
        return torch.from_numpy(pearson_correlation)
