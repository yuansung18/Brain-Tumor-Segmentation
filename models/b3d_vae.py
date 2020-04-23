import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PytorchModelBase
from .utils import get_tensor_from_array


class B3D_VAE(PytorchModelBase):

    def __init__(
            self,
            data_format: dict,
            batch_sampler_id: str = 'three_dim',
            floor_num: int = 3,
            kernel_size: int = 3,
            channel_num: int = 32,
            **kwargs,
    ):
        self.data_format = data_format
        self.kernel_size = kernel_size
        self.floor_num = floor_num
        super(B3D_VAE, self).__init__(
            batch_sampler_id=batch_sampler_id,
            data_format=data_format,
            forward_outcome_channels=channel_num,
            head_outcome_channels=channel_num,
            use_vae=True,
            **kwargs,
        )
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        for floor_idx in range(floor_num):
            channel_times = 2 ** floor_idx
            d = DownConv(channel_num * channel_times, kernel_size)
            self.down_layers.append(d)

        self.greenx2 = nn.Sequential(
            Green_block(channel_num * (2 ** floor_num), channel_num * (2 ** floor_num), kernel_size),
            Green_block(channel_num * (2 ** floor_num), channel_num * (2 ** floor_num), kernel_size)
        )

        for floor_idx in range(floor_num)[::-1]:
            channel_times = 2 ** floor_idx
            u = UpConv(channel_num * 2 * channel_times, kernel_size)
            self.up_layers.append(u)

        self.vae = VAE(channel_num * (2 ** floor_num), self.kernel_size, self.data_format)

    def forward_head(self, inp, data_idx):
        x = get_tensor_from_array(inp)
        x = self.heads[data_idx](x)

        return x

    def forward(self, x):
        x_out = [x]
        for i, down_layer in enumerate(self.down_layers):
            x = down_layer(x)
            if i == self.floor_num-1:
                x = self.greenx2(x)
            # print(x.shape)
            x_out.append(x)

        x_out = x_out[:-1]
        out_vae, vae_mean, vae_var = self.vae(x)
        for x_down, u in zip(x_out[::-1], self.up_layers):
            x = u(x, x_down)
        return x, out_vae, vae_mean, vae_var

    def build_heads(self, input_channels: list, output_channel: int):
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(input_channel, output_channel, kernel_size=1),
                nn.Dropout3d(p=0.1),
                Green_block(output_channel, output_channel, self.kernel_size)
            )
            for input_channel in input_channels
        ])

    def build_tails(self, input_channels, class_nums):
        return nn.ModuleList([
            nn.Conv3d(input_channels, class_num, kernel_size=1)
            for class_num in class_nums
        ])

    # def build_vaes(self, input_channel, class_nums):
    #     return nn.ModuleList([
    #         VAE(input_channel, self.kernel_size, self.data_format)
    #         for class_num in class_nums
    #     ])


class Green_block(nn.Module):
    """
    Green_block(in_ch, out_ch, kernel_size)
    ------------------------------------
    Implementation of the special residual block used in the paper. The block
    consists of two (GroupNorm --> ReLu --> 3x3x3 non-strided Convolution)
    units, with a residual connection from the input `x` to the output. Used
    internally in the model. Can be used independently as well.

    Parameters
    ----------
    `in_ch`: integer, required
        The input layer of this green block will have this no. of channels.
    `out_ch`: integer, required
        No. of filters to use in the 3D convolutional block. The output
        layer of this green block will have this no. of channels.
    `kernel_size`: int, required
        The kernel size of the 3D convolutional.
    """

    def __init__(self, in_ch, out_ch, kernel_size):
        super(Green_block, self).__init__()
        self.skip_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.drop1 = nn.Dropout3d(p=0.1)
        # self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.drop2 = nn.Dropout3d(p=0.1)
        # self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x_skip = self.skip_conv(x)
        # x = self.norm1(x)
        # x = F.leaky_relu(x)
        x = self.conv1(x)
        x = self.drop1(x)
        x = F.leaky_relu(x)
        # x = self.norm2(x)
        # x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = F.leaky_relu(x)
        x = x + x_skip

        return x


class DownConv(nn.Module):

    def __init__(self, in_ch, kernel_size):
        super(DownConv, self).__init__()
        out_ch = in_ch * 2
        self.down_conv = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size, padding=kernel_size // 2, stride=2)
        self.green1 = Green_block(in_ch, out_ch, kernel_size)
        self.green2 = Green_block(out_ch, out_ch, kernel_size)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.green1(x)
        x = self.green2(x)

        return x

class UpConv(nn.Module):

    def __init__(self, in_ch, kernel_size):
        super(UpConv, self).__init__()
        out_ch = in_ch // 2
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.green = Green_block(out_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x_up, x_down):
        x_up = self.conv1(x_up)
        x_up = self.upsampling(x_up)
        x = x_up + x_down
        x = self.green(x)

        return x


class VAE(nn.Module):

    def __init__(self, in_ch, kernel_size, data_format: dict):
        super(VAE, self).__init__()
        d, w, h = data_format['depth'], data_format['width'], data_format['height']
        self.shape = [d, h, w]
        self.VD = nn.ModuleDict({
            # 'GN': nn.GroupNorm(8, in_ch),
            'conv': nn.Conv3d(in_ch, 16, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            'drop': nn.Dropout3d(p=0.1),
            'dense': nn.Linear(16 * (d // 16) * (w // 16) * (h // 16), in_ch)
        })
        self.VDrew = nn.ModuleDict({
            'mean': nn.Linear(in_ch, in_ch // 2),
            'var': nn.Linear(in_ch, in_ch // 2)
        })
        self.VU = nn.ModuleDict({
            'dense': nn.Linear(in_ch // 2, (d // 16) * (w // 16) * (h // 16)),
            'conv1': nn.Conv3d(1, in_ch, kernel_size=1),
            'upsampling': nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        })
        self.VUp = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch // (2 ** factor), in_ch // (2 ** (factor + 1)), kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
                Green_block(in_ch // (2 ** (factor + 1)), in_ch // (2 ** (factor + 1)), kernel_size=kernel_size)
            )
            for factor in range(3)
        ])
        self.Vend = nn.Conv3d(in_ch // 8, data_format['class_num'], kernel_size=1)

    def forward(self, x):
        # VD Block (Reducing dimensionality of the data)
        # x = self.VD['GN'](x)
        # x = F.leaky_relu(x)
        x = self.VD['conv'](x)
        x = self.VD['drop'](x)
        x = F.leaky_relu(x)
        x = x.view(x.shape[0], -1)
        x = self.VD['dense'](x)

        # VDraw Block (Sampling)
        vae_mean = self.VDrew['mean'](x)
        vae_var = self.VDrew['var'](x)
        vae_std = torch.exp(0.5 * vae_var)
        x = vae_mean + vae_std * torch.randn_like(vae_std)

        # VU Block (Upsizing back to a depth of 256)
        x = self.VU['dense'](x)
        x = F.leaky_relu(x)
        x = x.view(x.shape[0], -1, self.shape[0] // 16, self.shape[1] // 16, self.shape[2] // 16)
        x = self.VU['conv1'](x)
        x = self.VU['upsampling'](x)

        # VUp Block
        for layer in self.VUp:
            x = layer(x)

        # Vend Block
        x = self.Vend(x)

        return x, vae_mean, vae_var
