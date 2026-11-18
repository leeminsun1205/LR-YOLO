"""S9-only frozen-memory checks, including the EMA copy and resume step."""
import json
from pathlib import Path

import torch
from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmyolo.registry import HOOKS

from .hooks import model_of

BANK_BUFFERS = ('prototype_vectors', 'prototype_modes', 'prototype_margins', 'prototype_reliability')


@HOOKS.register_module()
class PrototypeIntegrityHook(Hook):
    priority = 96

    def before_run(self, runner):
        self.initial = {name:getattr(model_of(runner), name).detach().cpu().clone()
                        for name in BANK_BUFFERS}

    def before_train(self, runner):
        if runner.cfg.sparse_contract['action'] in ('train','pilot'):
            if model_of(runner).prototype_warmup_updates != len(runner.train_dataloader):
                raise ValueError('Native loader length differs from the one-epoch memory warmup; '
                                 'inspect native filtering/batch settings before training')

    def after_train(self, runner):
        model = model_of(runner)
        candidates = [model]
        for hook in runner.hooks:
            if type(hook).__name__ == 'EMAHook':
                candidates.append(hook.ema_model.module)
        if len(candidates) != 2:
            raise AssertionError('S9 requires the original EMA hook')
        for candidate in candidates:
            if any(name in candidate.state_dict() for name in BANK_BUFFERS):
                raise AssertionError('Prototype buffers became persistent checkpoint weights')
            for name, value in self.initial.items():
                torch.testing.assert_close(getattr(candidate, name).cpu(), value, rtol=0, atol=0)
        report = dict(status='passed', frozen_bank_unchanged=True, ema_bank_unchanged=True,
            nonpersistent_bank=True, completed_epochs=runner.epoch, absolute_updates=runner.iter,
            sampling_step=model.sampling_step, warmup_updates=model.prototype_warmup_updates,
            identity_sha256=model.prototype_metadata['identity_sha256'],
            bank_sha256=model.prototype_metadata['bank_sha256'], auxiliary_trainable_parameters=0,
            semantic_status=model.prototype_metadata['quality']['status'],
            inactive_auxiliary_classes=model.prototype_metadata['quality']['inactive_auxiliary_classes'],
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total_parameters_including_frozen_lpips=sum(p.numel() for p in model.parameters()),
            bank_payload_bytes=sum(getattr(model,n).numel()*getattr(model,n).element_size()
                                   for n in BANK_BUFFERS))
        path = Path(runner.work_dir)/f'prototype_integrity_rank_{get_rank()}.json'
        path.write_text(json.dumps(report, indent=2)+'\n')
