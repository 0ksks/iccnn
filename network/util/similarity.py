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
        :param input: (batch, channel, pixel)
        :param return_type: (torch, numpy)
        :return: pearson_correlation: (channel, channel)
        """
        pretty_print("calculating pearson correlation... (network.util.similarity 22)")
        if isinstance(input, torch.Tensor):
            input = input.detach().cpu().numpy()
        _, channel, _ = input.shape
        mean_input = np.mean(input, axis=0, keepdims=True)  # (batch, channel, pixel)
        center_input: np.ndarray = input - mean_input  # (batch, channel, pixel)
        cov = np.mean(
            np.matmul(
                center_input,  # (batch, channel, pixel)
                center_input.transpose(0, 2, 1)  # (batch, pixel, channel)
            ),  # (batch, channel, channel)
            axis=0
        )  # (channel, channel)
        diag = np.diag(cov).reshape(channel, -1)  # (channel, 1)
        var = np.matmul(diag, diag.T)  # (channel, channel)
        pearson_correlation: np.ndarray = cov / (np.sqrt(var) + EPSILON)  # (channel, channel)
        if return_type == "numpy":
            return pearson_correlation
        return torch.from_numpy(pearson_correlation)

    @staticmethod
    def overlap(subset: torch.Tensor, superset: torch.Tensor) -> torch.Tensor:
        """
        :param subset: (channel, pixel)
        :param superset: (channel, pixel)
        :return: (sub_channel, super_channel)
        """
        if subset.shape[1] != superset.shape[1]:
            raise ValueError(
                f"Subset and superset must have the same number of pixels, "
                f"subset {subset.shape}, superset {superset.shape}"
            )

        intersection_area = torch.matmul(subset, superset.T)  # TODO may promote the model to enlarge the subset area,
        # TODO but we just want the intersected part more confident, under test
        difference_area = torch.matmul(subset, 1 - superset.T)

        return intersection_area / (difference_area + EPSILON)
