import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PytorchModelBase
from .utils import get_tensor_from_array, normalize_batch_image


class R2UNet(PytorchModelBase):

    def __init__(
        self,
        data_format: dict,
        device_id = 0,
        batch_sampler_id: str = 'three_dim',
        floor_num: int = 4,
        kernel_size: int = 3,
        channel_num: int = 16,
        conv_times: int = 2,
        use_position=False,
        dropout_rate: int = 0.,
        attention: bool = False,
        self_attention: int = 0,
        **kwargs,
    ):
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.conv_times = conv_times
        self.use_position = use_position
        self.device_id = device_id
        super(R2UNet, self).__init__(
            batch_sampler_id=batch_sampler_id,
            data_format=data_format,
            forward_outcome_channels=channel_num,
            head_outcome_channels=channel_num,
            **kwargs,
        )
        self.floor_num = floor_num
        image_chns = data_format['channels']
        if use_position:
            image_chns += 1
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.self_attention_module = nn.Sequential(*[
          SelfAttention(in_dim=2 ** floor_num * channel_num)
          for _ in range(self_attention)
        ])

        for floor_idx in range(floor_num):
            channel_times = 2 ** floor_idx
            d = DownConv(channel_num * channel_times, kernel_size, dropout_rate)
            self.down_layers.append(d)

        for floor_idx in range(floor_num)[::-1]:
            channel_times = 2 ** floor_idx
            up_conv_class = AttentionUpConv if attention else UpConv
            u = up_conv_class(
                channel_num * 2 * channel_times,
                kernel_size,
                dropout_rate,
            )
            self.up_layers.append(u)

    def forward_head(self, inp, data_idx):
        # inp, pos = inp['slice'], inp['position']
        x = get_tensor_from_array(inp, self.device_id)

        # if self.use_position:
        #     pos = get_tensor_from_array(pos)
        #     pos = pos.view(pos.shape[0], 1, 1, 1)
        #     pos = pos.expand(-1, 1, x.shape[-2], x.shape[-1])
        #     x = torch.cat([x, pos], dim=1)

        x = self.heads[data_idx](x)
        return x

    def forward(self, x):
        x_out = [x]
        for down_layer in self.down_layers:
            x = down_layer(x)
            x_out.append(x)

        x = self.self_attention_module(x)
        x_out = x_out[:-1]
        for x_down, u in zip(x_out[::-1], self.up_layers):
            x = u(x, x_down)
        return x

    def build_heads(self, input_channels: list, output_channel: int):
        if self.use_position:
            input_channels = [chn + 1 for chn in input_channels]
        return nn.ModuleList([
            RRCNN_block(
                input_channel,
                output_channel,
                kernel_size= self.kernel_size,
            )
            for input_channel in input_channels
        ])

    def build_tails(self, input_channels, class_nums):
        return nn.ModuleList([
            nn.Conv3d(input_channels, class_num, kernel_size=1)
            for class_num in class_nums
        ])

class RCNN_block(nn.Module):
    def __init__(self, ch_out, kernel_size, dropout_rate, t=2):
        super(RCNN_block, self).__init__()
        self.t = t
        self.conv = nn.Conv3d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.batch_norm = nn.BatchNorm3d(ch_out)
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
                if self.dropout.p == 0:
                    x1 = self.batch_norm(x1)
                else:
                    x1 = self.dropout(x1)
                x1 = F.relu(x1)

            x1 = self.conv(x+x1)
            x1 = self.batch_norm(x1)
            x1 = F.relu(x1)

        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, dropout_rate, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            RCNN_block(ch_out, kernel_size, dropout_rate, t=t),
            RCNN_block(ch_out, kernel_size, dropout_rate, t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class DownConv(nn.Module):

    def __init__(self, in_ch, kernel_size, dropout_rate):
        out_ch = in_ch * 2
        super(DownConv, self).__init__()
        self.down_conv = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size, padding=kernel_size//2, stride=2)
        self.batch_norm = nn.BatchNorm3d(in_ch)
        self.dropout = nn.Dropout3d(p=dropout_rate)
        self.conv = RRCNN_block(in_ch, out_ch, kernel_size, dropout_rate)

    def forward(self, x):
        x = self.down_conv(x)
        if self.dropout.p == 0:
            x = self.batch_norm(x)
        else:
            x = self.dropout(x)
        x = F.relu(x)
        x = conv(x)

        return x


class UpConv(nn.Module):

    def __init__(self, in_ch, kernel_size, dropout_rate):
        super(UpConv, self).__init__()
        out_ch = in_ch // 2
        self.conv_transpose = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2, stride=2)
        self.batch_norm = nn.BatchNorm3d(out_ch)
        self.conv = RRCNN_block(out_ch, out_ch, kernel_size=kernel_size)

    def forward(self, x_down, x_up):
        x_down = self.conv_transpose(x_down)
        x_down = self.batch_norm(x_down)
        x_down = F.relu(x_down)
        # print(x_down.shape, x_up.shape)
        if x_down.shape != x_up.shape:
            # this case will only happen when
            # x1 [N, C, D-1, H-1, W-1]
            # x2 [N, C, D,   H,   W  ]
            p_d = x_up.shape[2] - x_down.shape[2]
            p_h = x_up.shape[3] - x_down.shape[3]
            p_w = x_up.shape[4] - x_down.shape[4]
            pad = nn.ConstantPad3d((0, p_w, 0, p_h, 0, p_d), 0)
            x_down = pad(x_down)

        # x = torch.cat([x_down, x_up], dim=1)
        x = x_down + x_up
        # print(x.shape)
        
        x = self.conv(x)
        return x


class AttentionUpConv(nn.Module):

    def __init__(self, in_ch, kernel_size, conv_times, dropout_rate):
        super().__init__()
        out_ch = in_ch // 2
        self.conv_transpose = nn.ConvTranspose3d(
            in_ch,
            out_ch,
            kernel_size,
            padding=kernel_size // 2,
            stride=2,
        )
        self.query_conv = nn.Conv3d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=1,
        )
        self.key_conv = nn.Conv3d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=1,
        )
        self.attention_conv = nn.Conv3d(
            in_channels=out_ch,
            out_channels=1,
            kernel_size=1,
        )
        self.n_conv = RRCNN_block(
            in_ch, out_ch, kernel_size
        )

    def forward(self, x, x_down):
        x = self.conv_transpose(x)
        diff_z = x_down.size()[2] - x.size()[2]
        diff_x = x_down.size()[3] - x.size()[3]
        diff_y = x_down.size()[4] - x.size()[4]
        x = F.pad(x, [0, diff_x, 0, diff_y, 0, diff_z])

        q = self.query_conv(x)
        k = self.key_conv(x_down)
        attention = torch.sigmoid(self.attention_conv(F.relu(q + k)))
        value = x_down * attention
        concat = torch.cat([x, value], dim=1)
        out = self.n_conv(concat)
        return out


class SelfAttention(nn.Module):
    """
    Self attention Layer, adapted from
    https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X D X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, depth, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize,
            -1,
            width * height
        ).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, depth, width, height)

        out = self.gamma * out + x
        return out
