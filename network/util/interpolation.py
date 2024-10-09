from typing import Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate_grid(grid: np.ndarray, new_size: Union[int, tuple[int]]) -> np.ndarray:
    """
    对输入数组的最后两个维度进行插值，并根据new_size放大。
    无论输入为CHW或BCHW，都会保持前面的维度不变，只插值最后两个维度。

    :param grid: 输入的原始网格 (至少是二维numpy数组)
    :param new_size: 放大后的大小 (仅对最后两个维度有效)
    :return: 插值后的网格
    """
    # 获取输入数组的形状信息
    original_shape = grid.shape
    spatial_dims = original_shape[-2:]  # 最后两个维度 H 和 W
    other_dims = original_shape[:-2]  # 前面的维度 (例如C或B、C)

    # 新建一个空数组，保存插值后的结果
    if isinstance(new_size, tuple):
        new_size = new_size[0]
    interpolated = np.zeros((*other_dims, new_size, new_size))

    # 对所有前面的维度组合进行循环，分别插值
    for index in np.ndindex(*other_dims):
        # 创建插值函数
        x = np.linspace(0, spatial_dims[0] - 1, spatial_dims[0])
        y = np.linspace(0, spatial_dims[1] - 1, spatial_dims[1])
        interp_func = RegularGridInterpolator((x, y), grid[index], method='linear')

        # 新的插值网格
        x_new = np.linspace(0, spatial_dims[0] - 1, new_size)
        y_new = np.linspace(0, spatial_dims[1] - 1, new_size)
        x_new, y_new = np.meshgrid(x_new, y_new)
        points = np.array([x_new.flatten(), y_new.flatten()]).T

        # 对该通道进行插值，并放入插值结果数组
        interpolated[index] = interp_func(points).reshape(new_size, new_size)

    return interpolated
