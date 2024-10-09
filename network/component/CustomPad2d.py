from torch.nn import Module
import torch.nn as nn
import torch


# 用于重新实现二维填充的模块，替换过时的填充方法
class CustomPad2d(Module):
    def __init__(self, length):
        super(CustomPad2d, self).__init__()
        self.length = length
        self.zeroPad = nn.ZeroPad2d(self.length)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.shape
        output = self.zeroPad(input)

        for i in range(self.length):
            # 一层的四个切片
            output[:, :, self.length:self.length + h, i] = (
                output[:, :, self.length:self.length + h, self.length]
            )
            output[:, :, self.length:self.length + h, w + self.length + i] = (
                output[:, :, self.length:self.length + h, self.length - 1 + w]
            )
            output[:, :, i, self.length:self.length + w] = (
                output[:, :, self.length, self.length:self.length + w]
            )
            output[:, :, h + self.length + i, self.length:self.length + w] = (
                output[:, :, h + self.length - 1, self.length:self.length + w]
            )
        # 对角进行特别处理
        for j in range(self.length):
            for k in range(self.length):
                output[:, :, j, k] = output[:, :, self.length, self.length]
                output[:, :, j, w + self.length + k] = (
                    output[:, :, self.length, self.length - 1 + w]
                )
                output[:, :, h + self.length + j, k] = (
                    output[:, :, h + self.length - 1, self.length]
                )
                output[:, :, h + self.length + j, w + self.length + k] = (
                    output[:, :, h + self.length - 1, self.length - 1 + w]
                )
        return output


def overwrite_conv_2d(model: nn.Module):
    from network.component.CustomPad2d import CustomPad2d

    def custom_pad_2d_hook(module, input, output):
        custom_pad_2d = CustomPad2d(1)
        padded_input = custom_pad_2d(input[0])
        return module._conv_forward(padded_input, module.weight, module.bias)

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            layer.padding = (0, 0)
            layer.register_forward_hook(custom_pad_2d_hook)

    return model
