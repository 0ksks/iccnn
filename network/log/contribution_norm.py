import torch
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import wandb
from pytorch_lightning.loggers import Logger

from global_variable import get_config_value, parse_config_path, RUN_NAME


def get_contribution_norms(input_tensor: torch.Tensor, conv: torch.nn.Conv2d, device: torch.device) -> torch.Tensor:
    """
    :param input_tensor: (B, in_channel, H, W)
    :return: contribution_norms: (in_channel, out_channel)
    """
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        conv = conv.to(device)
        weight = conv.weight
        if conv.bias is not None:
            bias = conv.bias
        else:
            bias = None
        stride = conv.stride[0]
        padding = conv.padding[0]

        batch_size, in_channels, input_height, input_width = input_tensor.shape
        out_channels, _, kernel_height, kernel_width = weight.shape

        contribution_norms = torch.zeros((in_channels, out_channels)).to(device)

        for in_channel in range(in_channels):
            input_single_channel = input_tensor[:, in_channel:in_channel + 1, :, :]

            weight_single_channel = weight[:, in_channel:in_channel + 1, :, :]

            contributions = F.conv2d(input_single_channel, weight_single_channel, bias=None, stride=stride,
                                     padding=padding)

            contribution_norms[in_channel, :] = torch.norm(contributions, p='fro', dim=[2, 3])

        if bias is not None:
            contribution_norms += bias.view(1, -1)  # (1, out_channels) broadcast to (in_channels, out_channels)

        return contribution_norms


def draw_contribution_norms(contribution_norms: torch.Tensor, x_label: str, y_label: str) -> plt.Figure:
    contribution_norms = contribution_norms.cpu().float().numpy()
    fig, ax = plt.subplots()
    cax = ax.imshow(contribution_norms, cmap='gray', interpolation='nearest')
    fig.colorbar(cax)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def save_contribution_norms(fig: plt.Figure, logger: Logger, global_step: int):
    output_dir = parse_config_path(
        get_config_value("output.root") +
        get_config_value("output.filter_conv") +
        [RUN_NAME]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(str(output_dir), f"step_{global_step}.png")
    fig.savefig(save_path)
    logger.experiment.log({
        "Filter Contributions": [
            wandb.Image(
                save_path,
                caption="Filter Conv Kernel Norms"
            )
        ]
    })
