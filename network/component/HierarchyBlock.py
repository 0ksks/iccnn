import torch
import torch.nn as nn


class HierarchyBlock(nn.Module):
    def __init__(self, arch: list[int], num_classes: int, dropout=0.5):
        super(HierarchyBlock, self).__init__()
        in_channel_list = arch[:-1]
        out_channel_list = arch[1:]
        layers = []
        for in_channel, out_channel in zip(in_channel_list, out_channel_list):
            layers.append(nn.Linear(in_channel, out_channel))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(out_channel_list[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    B, C, H, W = 32, 512, 14, 14
    input_tensor = torch.rand(B, C, H, W)
    h_block = HierarchyBlock(arch=[H * W, 4, 6], num_classes=3, dropout=0.5)
    output = h_block(input_tensor.view(B, C, H * W))
    print(h_block)
    print(output.size())
