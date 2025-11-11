from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.backbones.csp_darknet import CSPLayer, Focus
from mmdet.utils import ConfigType, OptMultiConfig

from mmyolo.registry import MODELS
from ..layers import CSPLayerWithTwoConv, SPPFBottleneck
from ..utils import make_divisible, make_round
from .base_backbone import BaseBackbone


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


@MODELS.register_module()
class YOLOv8ReconstructCSPDarknet(BaseBackbone):
    """CSP-Darknet backbone used in YOLOv8.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5}.
            Defaults to P5.
        last_stage_out_channels (int): Final layer output channel.
            Defaults to 1024.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to: 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (Union[dict,list[dict]], optional): Initialization config
            dict. Defaults to None.

    Example:
        >>> from mmyolo.models import YOLOv8CSPDarknet
        >>> import torch
        >>> model = YOLOv8CSPDarknet()
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # the final out_channels will be set according to the param.
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)
        # Build upsampling path
        self.build_upsampling_path()

    def build_upsampling_path(self):
        """Build upsampling path for UNet-like structure."""
        # Get channel numbers from arch_settings
        channels = [make_divisible(setting[0], self.widen_factor) 
                   for setting in self.arch_settings['P5'][:3]]  # Only use first 3 stages
        
        # Create upsampling blocks
        self.up1 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1] * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        self.skip1 = nn.Conv2d(channels[1], channels[1], 1)
        self.fusion1 = SKFusion(channels[1])
        self.up_conv1 = CSPLayerWithTwoConv(
                channels[1],
                channels[1],
                num_blocks=2*2,
                add_identity=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[0] * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        self.skip2 = nn.Conv2d(channels[0], channels[0], 1)
        self.fusion2 = SKFusion(channels[0])
        self.up_conv2 = CSPLayerWithTwoConv(
                channels[0],
                channels[0],
                num_blocks=1*2,
                add_identity=True,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
                
        self.up3 = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        # Final output conv
        self.final_conv = ConvModule(
            channels[0],
            3,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    
    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = CSPLayerWithTwoConv(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs, features = [], []
        inputs = x
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            features.append(x)
            if i in self.out_indices:
                outs.append(x)
        
        # Get features for skip connections
        f1, f2, f3 = features[:3]  # Only use first 3 features
        # Decoder path with skip connections
        x = self.up1(f3)
        x = self.fusion1([x, self.skip1(f2)]) + x
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = self.fusion2([x, self.skip2(f1)]) + x
        x = self.up_conv2(x)
        
        x = self.up3(x)
        # Final conv to get original image size and channels
        out = self.final_conv(x)

        if return_features:
            return tuple(outs), out, features

        return tuple(outs), out