"""Post-neck adaptation of a released VOC LR-YOLO checkpoint.

This is NOT the author's training method. Only adapters learn, with detection
loss. Frozen modules remain in eval mode, including their BatchNorm buffers.
No LPIPS model/weights or reconstruction target are needed by this detector.
"""
import torch
from torch import nn
from mmengine.dist import get_rank

from mmyolo.models.detectors.yolo_detector import YOLODetector
from mmyolo.models.data_preprocessors.data_preprocessor import YOLOv5DetDataPreprocessor
from mmyolo.registry import MODELS


class ScaleShift(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, features):
        return self.scale * features + self.shift


class ResidualAdapter(nn.Module):
    def __init__(self, channels, reduction=16, gated=False):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.activation = nn.SiLU()
        self.up = nn.Conv2d(hidden, channels, 1)
        self.gated = gated
        if gated:
            self.gate = nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 1))
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        if self.gated:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features):
        residual = self.up(self.activation(self.down(features)))
        if self.gated:
            # Feature statistics are not an established estimate of illumination.
            # Detached features come from a frozen neck. Gate starts at one.
            stats = torch.stack((features.abs().mean((1, 2, 3)),
                                 features.std((1, 2, 3), unbiased=False)), dim=1)
            gate = 2 * torch.sigmoid(self.gate(torch.log1p(stats)))
            residual = residual * gate[:, :, None, None]
        return features + residual


@MODELS.register_module()
class VOCAdapterDarkPreprocessor(YOLOv5DetDataPreprocessor):
    """Author's gamma formula, training only; fixed dark validation is untouched."""
    def __init__(self, gamma_seed=42, **kwargs):
        super().__init__(**kwargs)
        self.gamma_seed = gamma_seed
        self._gamma_generators = {}

    def forward(self, data, training=False):
        result = super().forward(data, training=training)
        if training:
            inputs = result['inputs']
            key = str(inputs.device)
            if key not in self._gamma_generators:
                generator = torch.Generator(device=inputs.device)
                generator.manual_seed(self.gamma_seed + get_rank())
                self._gamma_generators[key] = generator
            gamma = 1.5 + torch.rand(inputs.shape[0], 1, 1, 1,
                                     device=inputs.device,
                                     generator=self._gamma_generators[key]) * 3.5
            result['inputs'] = inputs.pow(gamma).clamp(0, 1)
        return result


@MODELS.register_module()
class FrozenNeckAdapterDetector(YOLODetector):
    """Same VOC head/encoder/neck weights; optional trainable post-neck adapter.

    The reconstruction modules remain in state_dict for strict source-weight
    compatibility but are bypassed when decoder_free=True. Parameter memory
    therefore still includes those modules. No original source is patched.
    """
    def __init__(self, adapter_kind='none', reduction=16, decoder_free=True,
                 **kwargs):
        # Adapter has no BN; frozen modules do not need SyncBN conversion.
        kwargs['use_syncbn'] = False
        super().__init__(**kwargs)
        if adapter_kind not in ('none', 'scale_shift', 'residual', 'gated'):
            raise ValueError(adapter_kind)
        if reduction < 1:
            raise ValueError('reduction must be positive')
        self.adapter_kind = adapter_kind
        self.decoder_free = decoder_free
        channels = self.bbox_head.head_module.in_channels
        if adapter_kind == 'none':
            layers = [nn.Identity() for _ in channels]
        elif adapter_kind == 'scale_shift':
            layers = [ScaleShift(c) for c in channels]
        else:
            layers = [ResidualAdapter(c, reduction, adapter_kind == 'gated')
                      for c in channels]
        self.adapters = nn.ModuleList(layers)
        self._freeze_original()

    def _freeze_original(self):
        for module in (self.backbone, self.neck, self.bbox_head):
            module.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_original()
        return self

    def init_weights(self):
        super().init_weights()
        # Preserve identity even if framework initialization changes child layers.
        for module in self.adapters:
            if isinstance(module, ResidualAdapter):
                module.reset_identity()
            elif isinstance(module, ScaleShift):
                nn.init.ones_(module.scale)
                nn.init.zeros_(module.shift)

    def frozen_features(self, inputs, decoder_free=None):
        decoder_free = self.decoder_free if decoder_free is None else decoder_free
        with torch.no_grad():
            if decoder_free:
                outputs = []
                for index, name in enumerate(self.backbone.layers):
                    inputs = getattr(self.backbone, name)(inputs)
                    if index in self.backbone.out_indices:
                        outputs.append(inputs)
                features = tuple(outputs)
            else:
                features, _ = self.backbone(inputs)
            return self.neck(features)

    def extract_feat(self, batch_inputs):
        features = self.frozen_features(batch_inputs)
        if len(features) != len(self.adapters):
            raise RuntimeError('Feature levels do not match adapter levels')
        # Must stay outside no_grad: frozen head still backpropagates to adapters.
        return tuple(adapter(feature) for adapter, feature in zip(self.adapters, features))
