import torch


class EMAFeatureMap:
    """
    Exponential Moving Average Feature Map
    """

    def __init__(self, first_decay: float, decay: float, channel: int, feature_map_pixel: int, device: torch.device):
        self.device = device
        self.first_decay = torch.tensor(first_decay).to(dtype=torch.float, device=device)
        self.decay = torch.tensor(decay).to(dtype=torch.float, device=device)
        self.last_result = self._init_feature_map(channel, feature_map_pixel)
        self.is_first = True

    def _init_feature_map(self, channel: int, feature_map_pixel: int):
        return torch.zeros((channel, feature_map_pixel), dtype=torch.float).to(self.device).clone()

    def update(self, feature_map: torch.Tensor):
        """
        :param feature_map: (batch, channel, pixel)
        :return: EMA feature map: (channel, pixel)
        """
        decay = self.first_decay if self.is_first else self.decay
        self.last_result = decay * self.last_result + (1.0 - decay) * torch.mean(feature_map, dim=0)
        self.is_first = False
        return self.last_result
