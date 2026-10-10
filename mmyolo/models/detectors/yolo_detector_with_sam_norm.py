# Copyright (c) OpenMMLab. All rights reserved.
"""Parameter-space smoothing ported from the local mmyolo PGN implementation."""

import math

import torch
from mmengine.dist import get_world_size
from mmengine.optim import AmpOptimWrapper

from mmyolo.registry import MODELS
from .yolo_detector import YOLODetector


@MODELS.register_module()
class YOLODetectorSAMNorm(YOLODetector):
    """YOLO with a first-order approximation to loss gradient norm penalization.

    With ``g1 = grad(L(theta))`` and ``g2 = grad(L(theta + rho *
    g1 / ||g1||))``, update using ``(1 - alpha) * g1 + alpha * g2``.
    This approximates the gradient of ``L + alpha * rho * ||grad(L)||``
    locally, without building a second-order autograd graph. The target is
    the detection loss, not the output Jacobian.

    ``pgn_config`` accepts ``enable`` (False), ``rho`` (0.05), ``alpha``
    (0.8), and ``adaptive`` (False). The adaptive option preserves the
    original implementation's additional division of rho by the gradient
    norm. Enabled PGN supports single-process, full-precision training with
    no gradient accumulation. Use the ordinary model when PGN is disabled.
    """

    def __init__(self, *args, pgn_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        config = pgn_config or {}
        self.pgn_enable = config.get('enable', False)
        self.pgn_rho = float(config.get('rho', 0.05))
        self.pgn_alpha = float(config.get('alpha', 0.8))
        self.pgn_adaptive = config.get('adaptive', False)
        if not math.isfinite(self.pgn_rho) or self.pgn_rho < 0:
            raise ValueError('PGN rho must be finite and non-negative.')
        if not 0 <= self.pgn_alpha <= 1:
            raise ValueError('PGN alpha must be between 0 and 1.')
        if self.pgn_enable and get_world_size() > 1:
            raise RuntimeError(
                'PGN requires single-process training: the default MMEngine '
                'DDP wrapper bypasses this custom train_step.')

    def forward(self, inputs, gts=None, data_samples=None, mode='tensor'):
        # The haze preprocessor also supplies gts; ordinary detection uses
        # only the inputs and detection annotations, as in the original port.
        return super().forward(inputs, data_samples=data_samples, mode=mode)

    def train_step_pgn(self, data, optim_wrapper):
        """Compute two first-order gradients and update restored parameters."""
        if isinstance(optim_wrapper, AmpOptimWrapper):
            raise NotImplementedError('PGN does not support --amp.')
        if optim_wrapper._accumulative_counts != 1:
            raise NotImplementedError(
                'PGN requires optim_wrapper.accumulative_counts=1.')

        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            optim_wrapper.zero_grad()
            losses = self._run_forward(data, mode='loss')
            loss, log_vars = self.parse_losses(losses)
            # Count one optimizer iteration through the wrapper. The second
            # backward below belongs to this same iteration.
            optim_wrapper.backward(loss)
            params = dict(self.named_parameters())
            gradients_1 = {
                name: param.grad.detach().clone()
                for name, param in params.items() if param.grad is not None
            }
            if not gradients_1:
                raise RuntimeError('PGN found no parameter gradients.')
            grad_norm = torch.linalg.vector_norm(torch.stack([
                grad.float().norm(2) for grad in gradients_1.values()
            ]))
            rho = self.pgn_rho
            if self.pgn_adaptive:
                rho /= max(grad_norm.item(), 1e-12)

            old_params = {}
            try:
                with torch.no_grad():
                    for name, grad in gradients_1.items():
                        param = params[name]
                        old_params[name] = param.detach().clone()
                        param.add_(rho * grad / (grad_norm + 1e-12))

                optim_wrapper.zero_grad()
                losses_perturbed = self._run_forward(data, mode='loss')
                loss_perturbed, _ = self.parse_losses(losses_perturbed)
                loss_perturbed.backward()
                gradients_2 = {
                    name: param.grad.detach().clone()
                    for name, param in params.items()
                    if param.grad is not None
                }
            finally:
                # Restore weights even if the perturbed pass fails.
                with torch.no_grad():
                    for name, value in old_params.items():
                        params[name].copy_(value)

            optim_wrapper.zero_grad()
            for name, grad in gradients_1.items():
                if name in gradients_2:
                    params[name].grad = (
                        (1 - self.pgn_alpha) * grad
                        + self.pgn_alpha * gradients_2[name])
            optim_wrapper.step()
            optim_wrapper.zero_grad()
        return log_vars

    def train_step(self, data, optim_wrapper):
        if self.pgn_enable and self.training:
            return self.train_step_pgn(data, optim_wrapper)
        return super().train_step(data, optim_wrapper)
