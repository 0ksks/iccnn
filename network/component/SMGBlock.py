import torch
import torch.nn as nn

from global_variable import EPSILON
from network.util.EMA import EMAFeatureMap
from pretty_print import pretty_print


class SMGBlock(nn.Module):
    def __init__(
            self,
            decay: float,
            first_decay: float,
            channel: int,
            feature_map_pixel: int,
            device: torch.device
    ):
        """
        generate similarity
        """
        super(SMGBlock, self).__init__()
        self.device = device

        self.EMA_feature_map = EMAFeatureMap(
            decay=decay,
            first_decay=first_decay,
            channel=channel,
            feature_map_pixel=feature_map_pixel,
            device=self.device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (b, c, h, w)
        :return: correlation: (b, c, c)
        """
        pretty_print("similarity mask generating... (network.SMGBlock 32)")
        x = x.to(self.device)
        batch, channel, _, _ = x.size()
        x = x.view(batch, channel, -1).contiguous()  # b c p
        with torch.no_grad():
            f_mean = self.EMA_feature_map.update(x)  # c p
        _, pixel = f_mean.size()  # c p
        f_mean = f_mean.unsqueeze(0)  # 1 c p
        local = torch.matmul(
            x - f_mean,  # b c p
            (x - f_mean).transpose(1, 2)  # b p c
        )  # b c c
        diag = torch.eye(channel).unsqueeze(0).to(self.device)  # 1 c c
        cov = torch.sum(local * diag, dim=2).view(batch, channel, 1)  # b c 1
        norm = torch.sqrt(
            torch.matmul(
                cov,  # b c 1
                cov.transpose(1, 2)  # b 1 c
            )
        )  # b c c
        correlation = torch.div(local, norm + EPSILON)  # b c c
        return correlation
