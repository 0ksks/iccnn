import PIL
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import torchvision
import torchvision.transforms as transforms
import tqdm
import wandb
from pytorch_lightning.loggers import Logger

from global_variable import get_config_value, parse_config_path, RUN_NAME, EPSILON


def get_total_feature_map(layer_name, layer_instance, module, input, output):
    feature_map = output[0].detach().cpu()
    grid = torchvision.utils.make_grid(feature_map, normalize=True, scale_each=True)
    grid_img = grid.permute(1, 2, 0).numpy()
    return grid_img


def save_total_feature_map(layer_name, feature_maps: np.ndarray):
    """
    :param feature_maps: (h, w, c)
    """
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
    save_path = os.path.join(str(output_dir), f"step_{global_step}.webp")
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


def get_grouped_feature_map_mask(feature_maps: np.ndarray, group_mapping: np.ndarray) -> np.ndarray:
    """
    :param feature_maps: (h, w, c)
    :param group_mapping: (c, )
    :return: (group, h, w)
    """
    h, w, _ = feature_maps.shape
    group_indices = np.unique(group_mapping)
    masks = np.zeros((len(group_indices), h, w))
    for idx, group_index in enumerate(group_indices):
        current_group_index = group_mapping[group_mapping == group_index]
        current_feature_map = feature_maps[:, :, current_group_index]
        current_feature_map_min = current_feature_map.min()
        current_feature_map_max = current_feature_map.max()
        current_feature_map = (
                (current_feature_map - current_feature_map_min) /
                (current_feature_map_max - current_feature_map_min + EPSILON)
        )
        current_mask = 1 - current_feature_map.mean(axis=2)
        masks[idx] = current_mask
    return masks


def apply_grouped_feature_map_mask(
        origin_picture: torch.Tensor,
        masks: np.ndarray,
        global_step: int,
        logger: Logger,
        alpha: float = 0.5
) -> None:
    """
    :param origin_picture: (b, c, h, w) b=1
    :param masks: (group, h, w)
    :param alpha: [0, 1]
    """
    b, c, h, w = origin_picture.shape
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    denormalized_tensor = inv_normalize(origin_picture)
    group = masks.shape[0]
    masks = torch.from_numpy(masks).float().requires_grad_(False)

    fig_height = 5
    fig, axes = plt.subplots(1, group + 1, figsize=((group + 1) * fig_height, fig_height))
    fig.tight_layout(pad=4.0)
    original_image = denormalized_tensor.squeeze().permute(1, 2, 0).numpy()

    ax = axes[0]
    ax.imshow(original_image)
    ax.axis("off")
    for idx in range(group):
        mask = masks[idx]
        scale_h = h // masks.shape[1]
        scale_w = w // masks.shape[2]
        mask = mask.repeat_interleave(scale_h, dim=0).repeat_interleave(scale_w, dim=1).numpy()

        original_image = denormalized_tensor.squeeze().permute(1, 2, 0).numpy()

        ax = axes[idx + 1]
        ax.imshow(original_image)
        ax.imshow(mask, cmap="binary", alpha=alpha)  # 叠加mask的透明度效果
        ax.axis("off")

    output_dir = parse_config_path(
        get_config_value("output.root") +
        get_config_value("output.group_feature_maps") +
        [RUN_NAME]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(str(output_dir), f"step_{global_step}.webp")
    plt.savefig(save_path)
    plt.close(fig)

    logger.experiment.log({
        "Group Feature Maps": [
            wandb.Image(
                save_path,
                caption="Group Feature Maps"
            )
        ]
    })
