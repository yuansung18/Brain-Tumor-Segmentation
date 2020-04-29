import torch.nn as nn
import torch.nn.functional as F

from .base import PytorchModelBase
from .utils import get_tensor_from_array


class TWO_UNET(PytorchModelBase):

    def __init__(
            self,
            data_format: dict,
            first_channel_num: int = 16,
            second_channel_num: int = 32,
            kernel_size: int = 3,
            conv_time: int = 2,
            n_layer: int = 3,
            batch_sampler_id='three_dim',
            **kwargs,
    ):
        self.data_format = data_format
        self.kernel_size = kernel_size
        self.n_layer = n_layer
        super(TWO_UNET, self).__init__(
            batch_sampler_id=batch_sampler_id,
            data_format=data_format,
            head_outcome_channels=first_channel_num,
            forward_outcome_channels=second_channel_num,
            **kwargs,
        )
        # To work properly, kernel_size must be odd
        if kernel_size % 2 == 0:
            raise AssertionError('kernel_size({}) must be odd'.format(kernel_size))

        self.down1 = nn.ModuleList()
        self.up1 = nn.ModuleList()
        self.down2 = nn.ModuleList()
        self.conv_up2 = nn.ModuleList()
        self.inter_up2 = nn.ModuleList()

        # first stage u net
        self.head1 = nn.Sequential(
            nn.Conv3d(data_format['channels'], first_channel_num, kernel_size=kernel_size, padding=kernel_size // 3),
            Blue_block(first_channel_num, first_channel_num, kernel_size),
        )

        for i in range(n_layer):
            n_channel = (2 ** i) * first_channel_num
            down_conv = DownConv(n_channel, kernel_size, conv_time)
            self.down1.append(down_conv)

        self.convx2_1 = ConvNTimes(
            first_channel_num * (2 ** n_layer),
            first_channel_num * (2 ** n_layer),
            kernel_size,
            conv_time,
        )

        for i in range(n_layer):
            n_channel = (2 ** i) * first_channel_num
            up_conv = UpConv(n_channel * 2, n_channel, kernel_size, conv_time)
            self.up1.append(up_conv)

        self.tail1 = nn.Conv3d(first_channel_num, data_format['class_num'], kernel_size=1)

        # second stage u net
        self.head2 = nn.Sequential(
            nn.Conv3d(data_format['class_num'], second_channel_num, kernel_size=kernel_size, padding=kernel_size // 3),
            Blue_block(second_channel_num, second_channel_num, kernel_size),
        )

        for i in range(n_layer):
            n_channel = (2 ** i) * second_channel_num
            down_conv = DownConv(n_channel, kernel_size, conv_time)
            self.down2.append(down_conv)

        self.convx2_2 = ConvNTimes(
            second_channel_num * (2 ** n_layer),
            second_channel_num * (2 ** n_layer),
            kernel_size,
            conv_time,
        )

        for i in range(n_layer):
            n_channel = (2 ** i) * second_channel_num
            up_conv = UpConv(n_channel * 2, n_channel, kernel_size, conv_time)
            self.conv_up2.append(up_conv)

        self.tail2 = nn.Conv3d(second_channel_num, data_format['class_num'], kernel_size=1)

        for i in range(n_layer):
            n_channel = (2 ** i) * second_channel_num
            up_conv = InterUpConv(n_channel * 2, n_channel, kernel_size, conv_time)
            self.inter_up2.append(up_conv)

        self.tail3 = nn.Conv3d(second_channel_num, data_format['class_num'], kernel_size=1)

    def forward_head(self, inp):
        x = get_tensor_from_array(inp)
        if x.dim() != 5:
            raise AssertionError('input must have shape (batch_size, channel, D, H, W),\
                                 but get {}'.format(x.shape))
        return x

    def forward(self, x):
        x_inp = x
        x = self.head1(x_inp)
        x_out = [x]
        for i ,down_layer in enumerate(self.down1):
            x = down_layer(x)
            if i == self.n_layer-1:
                x = self.convx2_1(x)
            x_out.append(x)

        x_out = x_out[:-1]
        for x_down, u in zip(x_out[::-1], self.up1[::-1]):
            x = u(x, x_down)
        
        x = self.tail1(x)
        x_out1 = x
        x += x_inp
        x = self.head2(x)
        x_out = [x]
        for i ,down_layer in enumerate(self.down2):
            x = down_layer(x)
            if i == self.n_layer-1:
                x = self.convx2_2(x)
            x_out.append(x)

        x_out = x_out[:-1]
        x_inter = x
        for x_down, u in zip(x_out[::-1], self.inter_up2[::-1]):
            x_inter = u(x_inter, x_down)
        x_inter = self.tail3(x_inter)

        for x_down, u in zip(x_out[::-1], self.conv_up2[::-1]):
            x = u(x, x_down)
        x = self.tail2(x)

        return x_out1, x, x_inter


class Blue_block(nn.Module):
    """
    Blue_block(in_ch, out_ch, kernel_size)
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
        super(Blue_block, self).__init__()
        self.skip_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x_skip = self.skip_conv(x)
        x = self.norm1(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv2(x)
        x = x + x_skip

        return x
###########################################################
#             DnConv                                      #
#  input   [batch_num, input_channel,   D,   H,   W]      #
#  output  [batch_num, output_channel,  D/2, H/2, W/2]    #
###########################################################
class DownConv(nn.Module):

    def __init__(self, input_channel, kernel_size, conv_time):
        super(DownConv, self).__init__()
        output_channel = input_channel * 2
        self.down_conv = nn.Conv3d(
            input_channel, input_channel,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=2
        )
        self.conv_N_time = ConvNTimes(input_channel, output_channel, kernel_size, conv_time)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.conv_N_time(x)
        return x


###########################################################
#             UpConv                                      #
#  x1      [batch_num, x1_channel,    D/2, H/2, W/2]      #
#  x2      [batch_num, x2_channel,    D,   H,   W]        #
#  output  [batch_num, x2_channel,    D,   H,   W]      #
###########################################################
class UpConv(nn.Module):

    def __init__(self, x1_channel, x2_channel, kernel_size, conv_time):
        super(UpConv, self).__init__()
        self.up_conv = nn.ConvTranspose3d(
            x1_channel, x2_channel,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=2,
        )
        self.block = Blue_block(x2_channel, x2_channel, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        # print(x1.shape, x2.shape)
        if x1.shape != x2.shape:
            # this case will only happen when
            # x1 [N, C, D-1, H-1, W-1]
            # x2 [N, C, D,   H,   W  ]
            p_d = x2.shape[2] - x1.shape[2]
            p_h = x2.shape[3] - x1.shape[3]
            p_w = x2.shape[4] - x1.shape[4]
            pad = nn.ConstantPad3d((0, p_w, 0, p_h, 0, p_d), 0)
            x1 = pad(x1)

        x = x1 + x2
        x = self.block(x)
        return x

###########################################################
#             InterUpConv                                      #
#  x1      [batch_num, x1_channel,    D/2, H/2, W/2]      #
#  x2      [batch_num, x2_channel,    D,   H,   W]        #
#  output  [batch_num, x2_channel,    D,   H,   W]      #
###########################################################
class InterUpConv(nn.Module):

    def __init__(self, x1_channel, x2_channel, kernel_size, conv_time):
        super(InterUpConv, self).__init__()
        self.conv1 = nn.Conv3d(x1_channel, x2_channel, kernel_size=1)
        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.block = Blue_block(x2_channel, x2_channel, kernel_size)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.upsampling(x1)
        # print(x1.shape, x2.shape)
        if x1.shape != x2.shape:
            # this case will only happen when
            # x1 [N, C, D-1, H-1, W-1]
            # x2 [N, C, D,   H,   W  ]
            p_d = x2.shape[2] - x1.shape[2]
            p_h = x2.shape[3] - x1.shape[3]
            p_w = x2.shape[4] - x1.shape[4]
            pad = nn.ConstantPad3d((0, p_w, 0, p_h, 0, p_d), 0)
            x1 = pad(x1)

        x = x1 + x2
        x = self.block(x)
        return x


###########################################################
#             Conv_N_time                                 #
#  input   [batch_num, channel_num,   D,   H,   W]        #
#  output  [batch_num, channel_num,   D,   H,   W]        #
###########################################################
class ConvNTimes(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, conv_times):
        super(ConvNTimes, self).__init__()
        self.blocks = nn.ModuleList()

        for i in range(conv_times):
            if i == 0:
                block = Blue_block(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                )
            else:
                block = Blue_block(
                    out_ch,
                    out_ch,
                    kernel_size=kernel_size,
                )
            self.blocks.append(block)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x
