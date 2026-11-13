"""Audit the frozen-weight contract and record adapter-only run evidence."""
import hashlib
import json
from pathlib import Path
import time
import random

import numpy as np

import torch
from mmengine.dist import get_rank, get_world_size
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmyolo.registry import HOOKS


def tensor_fingerprint(state, include_adapters=True):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not include_adapters and name.startswith('adapters.'):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def unwrapped(model):
    return model.module if is_model_wrapper(model) else model


@HOOKS.register_module()
class VOCAdapterAuditHook(Hook):
    priority = 'NORMAL'  # Before LoggerHook resets CUDA peak counters.

    def before_train(self, runner):
        self.started = time.monotonic()
        self.peak_allocated = self.peak_reserved = 0
        model = unwrapped(runner.model)
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if not names or any(not n.startswith('adapters.') for n in names):
            raise RuntimeError('Only adapters may be trainable')
        if any(m.training for module in (model.backbone, model.neck, model.bbox_head)
               for m in module.modules()):
            raise RuntimeError('Frozen modules must stay in eval mode, including BN')
        self.original_hash = tensor_fingerprint(model.state_dict(), False)
        if not runner.cfg.resume:
            # Module construction consumes different RNG amounts for each case.
            # Reseed fresh-run data iteration after checkpoint/model initialization.
            seed = runner.cfg.randomness.seed + get_rank()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self._capture()

    def _capture(self):
        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._capture()
        for name, value in (outputs or {}).items():
            if 'loss' in name and not torch.isfinite(torch.as_tensor(value)).all():
                raise FloatingPointError(f'Non-finite {name}')

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._capture()

    def after_val_epoch(self, runner, metrics=None):
        self._capture()
        if get_rank() == 0:
            with (Path(runner.work_dir) / 'validation_history.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(epoch=runner.epoch, metrics=metrics)) + '\n')

    def before_save_checkpoint(self, runner, checkpoint):
        contract = dict(runner.cfg.adapter_contract)
        checkpoint.setdefault('meta', {})['voc_adapter'] = dict(
            case_id=contract['case_id'], base_checkpoint_sha256=contract['base_checkpoint_sha256'],
            training_contract=contract, frozen_state_sha256=self.original_hash,
            method='Post-neck adapter-only detection adaptation; not original LR training')

    def after_train(self, runner):
        self._capture()
        model = unwrapped(runner.model)
        unchanged = tensor_fingerprint(model.state_dict(), False) == self.original_hash
        if not unchanged:
            raise RuntimeError('Frozen parameters/buffers changed; invalidate this run')
        if get_rank() == 0:
            report = dict(elapsed_seconds=time.monotonic() - self.started,
                          peak_allocated_bytes=self.peak_allocated,
                          peak_reserved_bytes=self.peak_reserved,
                          memory_measurement='rank-zero persistent maxima before logger resets',
                          frozen_state_unchanged=unchanged, final_iteration=runner.iter,
                          gpu=torch.cuda.get_device_name(), world_size=get_world_size(),
                          cuda_version=torch.version.cuda, torch_version=torch.__version__,
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          total_parameters=sum(p.numel() for p in model.parameters()))
            (Path(runner.work_dir) / 'compute.json').write_text(json.dumps(report, indent=2))
