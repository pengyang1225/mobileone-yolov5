#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

# from .darknet import CSPDarknet
# from .network_blocks import BaseConv, CSPLayer, DWConv
import torch.nn as nn
from od.models.modules.common import Conv, Concat, C3,BaseConv,CSPLayer,DWConv
from utils.general import make_divisible

from alfred import logger, print_shape


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
            self,
            in_features=("stage2", "stage3", "stage4"),
            C3_size=256, C4_size=512, C5_size=1024,
            depthwise=False,
            version ='s'
    ):
        super().__init__()

        self.version = version
        gains = {
                'n': {'gd': 0.33, 'gw': 0.25},
                's': {'gd': 0.33, 'gw': 0.5},
                'm': {'gd': 0.67, 'gw': 0.75},
                'l': {'gd': 1, 'gw': 1},
                'x': {'gd': 1.33, 'gw': 1.25}
                }

        if self.version.lower() in gains:
            # only for yolov5
            depth= gains[self.version.lower()]['gd']  # depth gain
            width = gains[self.version.lower()]['gw']  # width gain
        else:
            depth = 0.33
            width = 0.5

        self.act = "silu"
        self.in_features = in_features
        self.in_channels = [ C3_size, C4_size, C5_size]
        self.out_shape =[]
        logger.info(
            "[YOLOPAFPN] in_feas: {}, in_channels: {}, with: {}".format(
                in_features, self.in_channels, width
            )
        )
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        if len(self.in_channels) == 4:
            width = 1
            self.lateral_conv0 = BaseConv(
                int(self.in_channels[3] * width), int(self.in_channels[2] * width), 1, 1, act=act
            )
            self.C3_p4 = CSPLayer(
                int(2 * self.in_channels[2] * width),
                int(self.in_channels[2] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=self.act,
            )  # cat

            self.reduce_conv1 = BaseConv(
                int(self.in_channels[2] * width), int(self.in_channels[1] * width), 1, 1, act=act
            )
            self.reduce_conv2 = BaseConv(
                int(self.in_channels[1] * width), int(self.in_channels[0] * width), 1, 1, act=act
            )
            self.C3_p2 = CSPLayer(
                int(2 * self.in_channels[0] * width),
                int(self.in_channels[0] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=self.act,
            )
            self.C3_p3 = CSPLayer(
                int(2 * self.in_channels[1] * width),
                int(self.in_channels[1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=self.act,
            )

            # bottom-up conv
            self.bu_conv1 = Conv(
                int(self.in_channels[2] * width), int(self.in_channels[2] * width), 3, 2, act=self.act
            )
            self.bu_conv2 = Conv(
                int(self.in_channels[1] * width), int(self.in_channels[1] * width), 3, 2, act=self.act
            )
            self.bu_conv3 = Conv(
                int(self.in_channels[0] * width), int(self.in_channels[0] * width), 3, 2, act=self.act
            )
            self.C3_n2 = CSPLayer(
                int(2 * self.in_channels[0] * width),
                int(self.in_channels[1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=self.act,
            )
            self.C3_n3 = CSPLayer(
                int(2 * self.in_channels[1] * width),
                int(self.in_channels[2] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=self.act,
            )
            self.C3_n4 = CSPLayer(
                int(2 * self.in_channels[2] * width),
                int(self.in_channels[3] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=self.act,
            )
            self.out_shape = {'P3_size': self.channels_outs[1],
                              'P4_size': self.channels_outs[1],
                              'P5_size': self.channels_outs[0]}
        else:
            width = 1
            self.lateral_conv0 = BaseConv(
                int(self.in_channels[2] * width), int(self.in_channels[1] * width), 1, 1, act=self.act
            )
            self.C3_p4 = CSPLayer(
                int(2 * self.in_channels[1] * width),
                int(self.in_channels[1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=self.act,
            )  # cat

            self.reduce_conv1 = BaseConv(
                int(self.in_channels[1] * width), int(self.in_channels[0] * width), 1, 1, act=self.act
            )
            self.C3_p3 = CSPLayer(
                int(2 * self.in_channels[0] * width),
                int(self.in_channels[0] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=self.act,
            )
            self.out_shape.append(int(self.in_channels[0] * width))

            # bottom-up conv
            self.bu_conv2 = Conv(
                int(self.in_channels[0] * width), int(self.in_channels[0] * width), 3, 2, act=self.act
            )
            self.C3_n3 = CSPLayer(
                int(2 * self.in_channels[0] * width),
                int(self.in_channels[1] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=self.act,
            )
            self.out_shape.append(self.in_channels[1] * width)

            # bottom-up conv
            self.bu_conv1 = Conv(
                int(self.in_channels[1] * width), int(self.in_channels[1] * width), 3, 2, act=self.act
            )
            self.C3_n4 = CSPLayer(
                int(2 * self.in_channels[1] * width),
                int(self.in_channels[2] * width),
                round(3 * depth),
                False,
                depthwise=depthwise,
                act=self.act,
            )
            self.out_shape.append(self.in_channels[2] * width)


    def forward(self, out_features):
        """
        Args:
            inputs: input images.
            out_features: output backbone features

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        # out_features = self.backbone(input)

        features = [out_features[f] for f in self.in_features]
        if len(features) == 3:
            [x2, x1, x0] = features
            # print_shape(x0)
            # print_shape(x2)

            fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
            f_out0 = self.upsample(fpn_out0)  # 512/16
            f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
            f_out0 = self.C3_p4(f_out0)  # 1024->512/16

            fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
            f_out1 = self.upsample(fpn_out1)  # 256/8
            f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
            pan_out2 = self.C3_p3(f_out1)  # 512->256/8

            p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
            p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
            pan_out1 = self.C3_n3(p_out1)  # 512->512/16

            p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
            p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
            pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

            outputs = (pan_out2, pan_out1, pan_out0)
            # for op3 in outputs:
            #     print_shape(op3)
            # 128,64,48 256,32,24 512,16,12
            return outputs
        else:
            [x3, x2, x1, x0] = features
            # print_shape(x0)
            # print_shape(x2)

            fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32

            f_out0 = self.upsample(fpn_out0)  # 512/16
            f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
            f_out0 = self.C3_p4(f_out0)  # 1024->512/16
            fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16

            f_out1 = self.upsample(fpn_out1)  # 256/8
            f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
            f_out2 = self.C3_p3(f_out1)  # 512->256/8
            fpn_out2 = self.reduce_conv2(f_out2)  # 512->256/16

            f_out2 = self.upsample(fpn_out2)  # 256/8
            f_out2 = torch.cat([f_out2, x3], 1)  # 256->512/8
            pan_out3 = self.C3_p2(f_out2)  # 512->256/8

            p_out2 = self.bu_conv3(pan_out3)  # 256->256/16
            p_out2 = torch.cat([p_out2, fpn_out2], 1)  # 256->512/16
            pan_out2 = self.C3_n2(p_out2)  # 512->512/16

            p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
            p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
            pan_out1 = self.C3_n3(p_out1)  # 512->512/16

            p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
            p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
            pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

            outputs = (pan_out3, pan_out2, pan_out1, pan_out0)
            # for op in outputs:
            #     print_shape(op)
            return outputs
