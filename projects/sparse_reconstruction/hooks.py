"""Cost, resume identity and sampler-step bookkeeping (no loss changes)."""
import hashlib
import json
from pathlib import Path
import time

import torch
from mmengine.dist import get_rank, get_world_size
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmyolo.registry import HOOKS


def model_of(runner):
    return runner.model.module if is_model_wrapper(runner.model) else runner.model


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@HOOKS.register_module()
class SparseRunHook(Hook):
    priority = 'NORMAL'  # Keep VRAM maxima before LoggerHook resets them.

    def before_run(self, runner):
        source = runner.cfg.get('load_from')
        if source:
            saved = torch.load(source, map_location='cpu', weights_only=False)
            state = {k.removeprefix('module.'): v for k,v in saved['state_dict'].items()}
            expected = model_of(runner).state_dict()
            if set(state) != set(expected):
                raise ValueError('Strict checkpoint keys mismatch: '
                    f'missing={sorted(set(expected)-set(state))}, extra={sorted(set(state)-set(expected))}')
            mismatch = [k for k in expected if state[k].shape != expected[k].shape]
            if mismatch:
                raise ValueError(f'Strict checkpoint shapes mismatch: {mismatch}')

    def before_train(self, runner):
        self.started = time.perf_counter()
        self.allocated = self.reserved = 0
        self.epoch_seconds = []
        self.frozen_before = frozen_hash(model_of(runner))
        self._capture()

    def _capture(self):
        self.allocated = max(self.allocated, torch.cuda.max_memory_allocated())
        self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())

    def before_train_epoch(self, runner):
        torch.cuda.synchronize()
        self.epoch_started = time.perf_counter()

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        model_of(runner).sampling_step = runner.iter

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._capture()
        for key, value in (outputs or {}).items():
            if 'loss' in key and not torch.isfinite(torch.as_tensor(value)).all():
                raise FloatingPointError(f'Nonfinite {key}')

    def after_train_epoch(self, runner):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.epoch_started
        self.epoch_seconds.append(seconds)
        self._capture()
        if get_rank() == 0:
            with (Path(runner.work_dir) / 'epoch_times.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(epoch=runner.epoch+1, train_seconds=seconds,
                    global_images_per_second=len(runner.train_dataloader)*
                    runner.cfg.train_dataloader.batch_size*get_world_size()/seconds))+'\n')

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self._capture()

    def after_val_epoch(self, runner, metrics=None):
        self._capture()
        if get_rank() == 0:
            with (Path(runner.work_dir) / 'validation_history.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(epoch=runner.epoch, metrics=metrics))+'\n')

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint.setdefault('meta', {})['sparse_reconstruction'] = dict(
            contract=dict(runner.cfg.sparse_contract),
            stage=runner.cfg.sparse_contract['stage'],
            case_id=runner.cfg.sparse_contract['case_id'],
            initialization=runner.cfg.get('sparse_initialization'),
            sampling='derived from seed/rank/absolute runner.iter; no global RNG consumption',
            resume_randomness='optimizer/loop restored; data/gamma RNG stream not bitwise restored')

    def after_train(self, runner):
        torch.cuda.synchronize()
        self._capture()
        if frozen_hash(model_of(runner)) != self.frozen_before:
            raise RuntimeError('Frozen weights changed; invalidate run')
        report = dict(elapsed_seconds=time.perf_counter()-self.started,
            train_epoch_seconds=self.epoch_seconds, peak_allocated_bytes=self.allocated,
            peak_reserved_bytes=self.reserved, rank=get_rank(), world_size=get_world_size(),
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            cuda_version=torch.version.cuda, frozen_weights_unchanged=True,
            completed_epochs=runner.epoch, contract=dict(runner.cfg.sparse_contract),
            source_initialization=runner.cfg.get('sparse_initialization'),
            memory_scope='whole training/validation session, not sum across ranks',
            randomness_limit='resume is not bitwise identical to uninterrupted dataloader/gamma RNG')
        (Path(runner.work_dir)/f'compute_rank_{get_rank()}.json').write_text(json.dumps(report, indent=2))


@HOOKS.register_module()
class SparsePostValidationCheckpointHook(Hook):
    """Atomic rolling resume slot AFTER best-score bookkeeping; before budget stop."""
    priority = 95

    def after_val_epoch(self, runner, metrics=None):
        if get_rank() == 0:
            folder = Path(runner.work_dir)
            destination = folder/'latest_resume.pth'
            runner.message_hub.update_info('last_ckpt', str(destination))
            runner.save_checkpoint(str(folder), 'latest_resume.pending.pth',
                save_optimizer=True, save_param_scheduler=False,
                meta=dict(epoch=runner.epoch, iter=runner.iter))
            (folder/'latest_resume.pending.pth').replace(destination)
            (folder/'last_checkpoint').write_text(str(destination))
        if get_world_size() > 1:
            torch.distributed.barrier()
