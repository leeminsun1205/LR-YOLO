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


@MODELS.register_module()
class YOLOv8ReconstructCSPDarknet_Original(BaseBackbone):
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
        """Build upsampling path based on the PANet-like diagram."""
        # Get channel numbers from arch_settings
        # f1 (stem out, H/2)
        c1 = make_divisible(self.arch_settings['P5'][0][0], self.widen_factor)
        # f2 (stage1 out, H/4)
        c2 = make_divisible(self.arch_settings['P5'][0][1], self.widen_factor)
        # f3 (stage2 out, H/8)
        c3 = make_divisible(self.arch_settings['P5'][1][1], self.widen_factor)

        # Number of blocks for each CSPLayer
        n_blocks = make_round(3, self.deepen_factor)

        # --- 1. Top-Down Path (FPN) ---
        # P3_td (from F3, H/8, c3)
        self.pan_c3_td = CSPLayerWithTwoConv(
            c3, c3, num_blocks=n_blocks, add_identity=True,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # Upsample c3 -> c2 (H/8 -> H/4)
        self.pan_up_32 = nn.Sequential(
            nn.Conv2d(c3, c2 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        self.pan_lat_f2 = nn.Conv2d(c2, c2, 1)  # 1x1 for F2 (H/4, c2)

        # P2_td (H/4, c2)
        self.pan_c2_td = CSPLayerWithTwoConv(
            c2, c2, num_blocks=n_blocks, add_identity=True,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # Upsample c2 -> c1 (H/4 -> H/2)
        self.pan_up_21 = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        self.pan_lat_f1 = nn.Conv2d(c1, c1, 1)  # 1x1 for F1 (H/2, c1)

        # P1_td (H/2, c1)
        self.pan_c1_td = CSPLayerWithTwoConv(
            c1, c1, num_blocks=n_blocks, add_identity=True,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # --- 2. Bottom-Up Path (PAN) ---
        # Downsample c1 -> c2 (H/2 -> H/4)
        self.pan_down_12 = ConvModule(
            c1, c2, kernel_size=3, stride=2, padding=1,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # P2_bu (H/4, c2)
        self.pan_c2_bu = CSPLayerWithTwoConv(
            c2, c2, num_blocks=n_blocks, add_identity=True,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # Downsample c2 -> c3 (H/4 -> H/8)
        self.pan_down_23 = ConvModule(
            c2, c3, kernel_size=3, stride=2, padding=1,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # P3_bu (H/8, c3)
        self.pan_c3_bu = CSPLayerWithTwoConv(
            c3, c3, num_blocks=n_blocks, add_identity=True,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # --- 3. Final Top-Down Path (FPN) ---
        # Upsample c3 -> c2 (H/8 -> H/4)
        self.pan_up_32_final = nn.Sequential(
            nn.Conv2d(c3, c2 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

        # P2_final (H/4, c2)
        self.pan_c2_final = CSPLayerWithTwoConv(
            c2, c2, num_blocks=n_blocks, add_identity=True,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # Upsample c2 -> c1 (H/4 -> H/2)
        self.pan_up_21_final = nn.Sequential(
            nn.Conv2d(c2, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

        # P1_final (H/2, c1)
        self.pan_c1_final = CSPLayerWithTwoConv(
            c1, c1, num_blocks=n_blocks, add_identity=True,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # --- Final Output Conv ---
        # Upsample H/2 -> H
        self.final_up = nn.Sequential(
            nn.Conv2d(c1, c1 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        # Final conv to 3 channels
        self.final_conv = ConvModule(
            c1, 3, kernel_size=3, stride=1, padding=1,
            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

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
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            features.append(x)
            if i in self.out_indices:
                outs.append(x)

        # Get features for restoration head (PANet)
        # f1: stem out (H/2, c1)
        # f2: stage1 out (H/4, c2)
        # f3: stage2 out (H/8, c3)
        f1, f2, f3 = features[0], features[1], features[2]

        # --- Restoration Head (PANet) ---

        # --- 1. Top-Down Path (FPN) ---
        # P3_td (from F3)
        p3_td = self.pan_c3_td(f3)

        # P2_td (from F2 and P3_td)
        p2_in = self.pan_up_32(p3_td) + self.pan_lat_f2(f2)
        p2_td = self.pan_c2_td(p2_in)

        # P1_td (from F1 and P2_td)
        p1_in = self.pan_up_21(p2_td) + self.pan_lat_f1(f1)
        p1_td = self.pan_c1_td(p1_in)

        # --- 2. Bottom-Up Path (PAN) ---
        # P2_bu (from P1_td and P2_td)
        p2_bu_in = self.pan_down_12(p1_td) + p2_td
        p2_bu = self.pan_c2_bu(p2_bu_in)

        # P3_bu (from P2_bu and P3_td)
        p3_bu_in = self.pan_down_23(p2_bu) + p3_td
        p3_bu = self.pan_c3_bu(p3_bu_in)

        # --- 3. Final Top-Down Path (FPN) ---
        # P2_final (from P3_bu and P2_bu)
        p2_final_in = self.pan_up_32_final(p3_bu) + p2_bu
        p2_final = self.pan_c2_final(p2_final_in)

        # P1_final (from P2_final and P1_td)
        p1_final_in = self.pan_up_21_final(p2_final) + p1_td
        p1_final = self.pan_c1_final(p1_final_in)

        # --- Final Output ---
        x_up = self.final_up(p1_final)
        out = self.final_conv(x_up)  # Restored Image

        if return_features:
            # 'outs' contains P3,P4,P5 features for the detector
            # 'out' is the restored image
            # 'features' contains all intermediate backbone features
            return tuple(outs), out, features

        # 'outs' = YOLO detector features (P3, P4, P5)
        # 'out' = Restored Image
        return tuple(outs), out