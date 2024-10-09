import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torchvision
import torchvision.transforms as transforms
import tqdm
import wandb
from pytorch_lightning.loggers import Logger

from global_variable import get_config_value, parse_config_path, RUN_NAME


def get_total_feature_map(layer_name, layer_instance, module, input, output):
    feature_map = output[0].detach().cpu()
    grid = torchvision.utils.make_grid(feature_map, normalize=True, scale_each=True)
    grid_img = grid.permute(1, 2, 0).numpy()
    return grid_img


def save_total_feature_map(layer_name, feature_maps: np.ndarray):
    layer_name = layer_name.replace('.', '_')
    output_dir = parse_config_path(
        get_config_value("output.root") +
        get_config_value("output.feature_maps") +
        [RUN_NAME, layer_name]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filter_idx in tqdm.tqdm(
            range(feature_maps.shape[2]),
            desc='saving feature map',
            dynamic_ncols=True,
    ):
        plt.imshow(feature_maps[:, :, filter_idx], cmap="gray")
        plt.axis('off')
        plt.savefig(os.path.join(str(output_dir), f"filter_{filter_idx}.webp"))


def get_sample_feature_map(layer_name, layer_instance, module, input, output) -> dict:
    feature_map = output[0].detach().cpu()
    grid = torchvision.utils.make_grid(feature_map, normalize=True, scale_each=True)
    grid_img = grid.permute(1, 2, 0).numpy()

    return {layer_name: grid_img}


def save_sample_feature_maps(
        origin_picture: torch.Tensor,
        feature_maps: dict[str, np.ndarray],
        logger: Logger,
        global_step: int,
        num_cols: int
) -> None:
    num_layers = len(feature_maps) + 1
    num_rows = (num_layers + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    fig.tight_layout(pad=4.0)

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    denormalized_tensor = inv_normalize(origin_picture)

    to_pil = transforms.ToPILImage()
    image = to_pil(denormalized_tensor.squeeze(0))
    ax = axes[0, 0]
    ax.imshow(image)
    ax.set_title("origin", fontsize=32)
    ax.axis("off")

    for idx, (layer_name, feature_map) in enumerate(feature_maps.items(), 1):
        row = idx // num_cols
        col = idx % num_cols

        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.imshow(feature_map[:, :, 0], cmap="gray")
        ax.set_title(layer_name, fontsize=32)
        ax.axis("off")

    for idx in range(num_layers, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis("off")

    output_dir = parse_config_path(
        get_config_value("output.root") +
        get_config_value("output.feature_maps") +
        [RUN_NAME]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(str(output_dir), f"feature_map_step_{global_step}.webp")
    plt.savefig(save_path)
    plt.close(fig)

    logger.experiment.log({
        "Feature Maps": [
            wandb.Image(
                save_path,
                caption="Feature Maps"
            )
        ]
    })
