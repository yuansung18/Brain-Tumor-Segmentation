import torch.nn as nn
import torch.nn.functional as F

from .base import PytorchModelBase
from .utils import get_tensor_from_array


class TWO_VNET(PytorchModelBase):

    def __init__(
            self,
            data_format: dict,
            first_channel_num: int = 16,
            second_channel_num: int = 32,
            kernel_size: int = 3,
            conv_time: int = 2,
            n_layer: int = 4,
            batch_sampler_id='three_dim',
            dropout_rate: float = 0.,
            **kwargs,
    ):
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        super(TWO_VNET, self).__init__(
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

        # first stage v net
        self.head1 = Duplicate(data_format['channels'], first_channel_num, kernel_size, dropout_rate)

        for i in range(n_layer):
            n_channel = (2 ** i) * first_channel_num
            down_conv = DownConv(n_channel, kernel_size, conv_time, dropout_rate)
            self.down1.append(down_conv)

        for i in range(n_layer):
            n_channel = (2 ** i) * first_channel_num
            up_conv = UpConv(n_channel * 2, n_channel, kernel_size, conv_time, dropout_rate)
            self.up1.append(up_conv)

        self.tail1 = nn.Conv3d(first_channel_num, data_format['class_num'], kernel_size=1)

        # second stage u net
        self.head2 = Duplicate(data_format['channels'], second_channel_num, kernel_size, dropout_rate)

        for i in range(n_layer):
            n_channel = (2 ** i) * second_channel_num
            down_conv = DownConv(n_channel, kernel_size, conv_time, dropout_rate)
            self.down2.append(down_conv)

        for i in range(n_layer):
            n_channel = (2 ** i) * second_channel_num
            up_conv = UpConv(n_channel * 2, n_channel, kernel_size, conv_time, dropout_rate)
            self.conv_up2.append(up_conv)

        self.tail2 = nn.Conv3d(second_channel_num, data_format['class_num'], kernel_size=1)

        for i in range(n_layer):
            n_channel = (2 ** i) * second_channel_num
            up_conv = InterUpConv(n_channel * 2, n_channel, kernel_size, conv_time, dropout_rate)
            self.inter_up2.append(up_conv)

        self.tail3 = nn.Conv3d(second_channel_num, data_format['class_num'], kernel_size=1)

    def forward_head(self, inp, data_idx):
        x = get_tensor_from_array(inp)
        if x.dim() != 5:
            raise AssertionError('input must have shape (batch_size, channel, D, H, W),\
                                 but get {}'.format(x.shape))
        return x

    def forward(self, x):
        x_inp = x
        x = self.head1(x_inp)
        x_out = [x]
        for i, down_layer in enumerate(self.down1):
            x = down_layer(x)
            x_out.append(x)

        x_out = x_out[:-1]
        for x_down, u in zip(x_out[::-1], self.up1[::-1]):
            x = u(x, x_down)

        x = self.tail1(x)
        x_out1 = x
        x += x_inp
        x = self.head2(x)
        x_out = [x]
        for i, down_layer in enumerate(self.down2):
            x = down_layer(x)
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



###########################################################
#             DnConv                                      #
#  input   [batch_num, input_channel,   D,   H,   W]      #
#  output  [batch_num, output_channel,  D/2, H/2, W/2]    #
###########################################################
class DownConv(nn.Module):

    def __init__(self, input_channel, kernel_size, conv_time, dropout_rate):
        super(DownConv, self).__init__()
        output_channel = input_channel * 2
        self.down_conv = nn.Conv3d(input_channel, output_channel, kernel_size=kernel_size, stride=2)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.batch_norm = nn.BatchNorm3d(output_channel)
        self.conv_N_time = ConvNTimes(output_channel, kernel_size, conv_time, dropout_rate)

    def forward(self, x):
        x = self.down_conv(x)
        if self.dropout.p == 0:
            x = self.batch_norm(x)
        else:
            x = self.dropout(x)
        x = F.relu(x)
        x = self.conv_N_time(x)
        return x


###########################################################
#             UpConv                                      #
#  x1      [batch_num, x1_channel,    D/2, H/2, W/2]      #
#  x2      [batch_num, x2_channel,    D,   H,   W]        #
#  output  [batch_num, x2_channel*2,  D*2, H*2, W*2]      #
###########################################################
class UpConv(nn.Module):

    def __init__(self, x1_channel, x2_channel, kernel_size, conv_time, dropout_rate):
        super(UpConv, self).__init__()
        self.up_conv = nn.ConvTranspose3d(x1_channel, x2_channel, kernel_size=kernel_size, stride=2)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.batch_norm = nn.BatchNorm3d(x2_channel)
        self.conv_N_time = ConvNTimes(x2_channel, kernel_size, conv_time, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        if self.dropout.p == 0:
            x1 = self.batch_norm(x1)
        else:
            x1 = self.dropout(x1)
        x1 = F.relu(x1)
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

        # x = torch.cat((x1, x2), 1)
        x = x1 + x2
        x = self.conv_N_time(x)
        return x

###########################################################
#             InterUpConv                                      #
#  x1      [batch_num, x1_channel,    D/2, H/2, W/2]      #
#  x2      [batch_num, x2_channel,    D,   H,   W]        #
#  output  [batch_num, x2_channel*2,  D*2, H*2, W*2]      #
###########################################################
class InterUpConv(nn.Module):

    def __init__(self, x1_channel, x2_channel, kernel_size, conv_time, dropout_rate):
        super(InterUpConv, self).__init__()
        self.conv = nn.Conv3d(x1_channel, x2_channel, kernel_size=1)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.batch_norm = nn.BatchNorm3d(x2_channel)
        self.upsampling = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv_N_time = ConvNTimes(x2_channel, kernel_size, conv_time, dropout_rate)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        if self.dropout.p == 0:
            x1 = self.batch_norm(x1)
        else:
            x1 = self.dropout(x1)
        x1 = F.relu(x1)
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

        # x = torch.cat((x1, x2), 1)
        x = x1 + x2
        x = self.conv_N_time(x)
        return x


###########################################################
#             Conv_N_time                                 #
#  input   [batch_num, channel_num,   D,   H,   W]        #
#  output  [batch_num, channel_num,   D,   H,   W]        #
###########################################################
class ConvNTimes(nn.Module):

    def __init__(self, channel_num, kernel_size, N, dropout_rate):
        super(ConvNTimes, self).__init__()

        self.convs = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.dropout = nn.Dropout3d(p=dropout_rate)

        for _ in range(N):
            conv = nn.Conv3d(
                channel_num,
                channel_num,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.convs.append(conv)
            norm = nn.BatchNorm3d(channel_num)
            self.batchnorms.append(norm)

    def forward(self, x):
        for conv, batchnorm in zip(self.convs, self.batchnorms):
            x = conv(x)
            if self.dropout.p == 0:
                x = batchnorm(x)
            else:
                x = self.dropout(x)
            x = F.relu(x)
        return x


###########################################################
#             Duplication                                 #
#  input   [batch_num, input_channel,    D,   H,   W]     #
#  output  [batch_num, first_channel_num,  D,   H,   W]     #
###########################################################
class Duplicate(nn.Module):

    def __init__(self, input_channel, first_channel_num, kernel_size, dropout_rate):
        super(Duplicate, self).__init__()
        self.duplicate = nn.Conv3d(
            input_channel,
            first_channel_num,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.batch_norm = nn.BatchNorm3d(first_channel_num)

    def forward(self, inp):
        x = self.duplicate(inp)
        if self.dropout.p == 0:
            x = self.batch_norm(x)
        else:
            x = self.dropout(x)
        x = F.relu(x)
        return x
