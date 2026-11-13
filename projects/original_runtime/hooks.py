"""Measure an original clean/dark epoch, preserving original training behavior."""
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
from mmengine.dist import get_rank, get_world_size
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmyolo.registry import HOOKS


@HOOKS.register_module()
class OriginalEpochTimingHook(Hook):
    priority = 'NORMAL'  # Capture peaks before LoggerHook resets them.

    def __init__(self, warmup_steps=20):
        self.warmup_steps = warmup_steps

    def sync_time(self):
        torch.cuda.synchronize()
        return time.perf_counter()

    def capture_memory(self):
        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())

    def before_train(self, runner):
        from mmengine.runner.loops import EpochBasedTrainLoop
        from mmengine.runner.utils import calc_dynamic_intervals
        self.started = self.sync_time()
        self.initialization_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self.peak_allocated = self.peak_reserved = 0
        self.steps = []
        self.train_seconds = self.val_seconds = 0.
        self.regular_checkpoint_seconds = self.best_checkpoint_seconds = 0.
        self.steady_started = None
        self.loss_keys = []
        self.checkpoint_save_calls = 0
        model = runner.model.module if is_model_wrapper(runner.model) else runner.model
        self.trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        if any(n.startswith('adapters.') for n in self.trainable):
            raise RuntimeError('Original benchmark must not include adapters')
        if any(p.requires_grad for p in model.criterion_lpips.parameters()):
            raise RuntimeError('Original LPIPS parameters must remain frozen')
        stage = runner.cfg.original_runtime['stage']
        decoder_parameters = [p for module in (model.neck, model.bbox_head) for p in module.parameters()]
        if stage == 'clean' and not all(p.requires_grad for p in decoder_parameters):
            raise RuntimeError('Original clean stage must train neck/head parameters')
        if stage == 'dark' and any(p.requires_grad for p in decoder_parameters):
            raise RuntimeError('Original dark stage must freeze neck/head parameters')
        if not any(p.requires_grad for p in model.backbone.parameters()):
            raise RuntimeError('Original encoder must remain trainable')
        self.parameter_counts = dict(
            total=sum(p.numel() for p in model.parameters()),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            encoder_neck_head_trainable={name: sum(p.numel() for p in module.parameters() if p.requires_grad)
                                        for name, module in [('backbone', model.backbone),
                                                             ('neck', model.neck), ('head', model.bbox_head)]})
        # Ask the actual installed loop for the source 100-epoch validation schedule.
        original = runner.cfg.original_runtime['source_train_cfg']
        milestones, intervals = calc_dynamic_intervals(
            original['val_interval'], [tuple(v) for v in original.get('dynamic_intervals', [])])
        self.original_validation_epochs = []
        for completed in range(1, original['max_epochs'] + 1):
            probe = SimpleNamespace(epoch=completed, dynamic_milestones=milestones,
                                    dynamic_intervals=intervals)
            EpochBasedTrainLoop._decide_current_val_interval(probe)
            if completed >= original.get('val_begin', 1) and (
                    completed % probe.val_interval == 0 or completed == original['max_epochs']):
                self.original_validation_epochs.append(completed)
        self.capture_memory()

    def before_train_epoch(self, runner):
        if get_world_size() > 1:
            torch.distributed.barrier()
        self.train_started = self.sync_time()

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.step_started = self.sync_time()
        if batch_idx == self.warmup_steps:
            self.steady_started = self.step_started

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.steps.append(self.sync_time() - self.step_started)
        self.capture_memory()
        for name, value in (outputs or {}).items():
            if 'loss' in name and not math.isfinite(float(value)):
                raise FloatingPointError(f'Non-finite {name} in original probe')
        if batch_idx == 0:
            self.loss_keys = sorted(outputs or {})
            required = {'one2many_loss_cls', 'one2many_loss_bbox', 'one2many_loss_dfl',
                        'one2one_loss_cls', 'one2one_loss_bbox', 'one2one_loss_dfl',
                        'loss_reconstruct_l1', 'loss_reconstruct_contrast'}
            if not required.issubset(self.loss_keys):
                raise RuntimeError(f'Original detection/L1/LPIPS losses missing: {self.loss_keys}')

    def after_train_epoch(self, runner):
        self.train_ended = self.sync_time()
        self.train_seconds = self.train_ended - self.train_started
        self.steady_epoch_seconds = None
        if self.steady_started is not None and len(self.steps) > self.warmup_steps:
            self.steady_epoch_seconds = ((self.train_ended - self.steady_started)
                                         * len(self.steps) / (len(self.steps) - self.warmup_steps))
        self.capture_memory()

    def before_val_epoch(self, runner):
        if get_world_size() > 1:
            torch.distributed.barrier()
        self.val_started = self.sync_time()

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.capture_memory()

    def after_val_epoch(self, runner, metrics=None):
        self.val_ended = self.sync_time()
        self.val_seconds = self.val_ended - self.val_started
        self.metrics = metrics
        self.capture_memory()

    def before_save_checkpoint(self, runner, checkpoint):
        self.checkpoint_save_calls += 1
        checkpoint.setdefault('meta', {})['original_runtime_probe'] = dict(
            scientific_result=False, stage=runner.cfg.original_runtime['stage'],
            probe_epochs=1, original_schedule_epochs=100,
            initialization=runner.cfg.original_runtime['initialization'])

    def after_train(self, runner):
        finished = self.sync_time()
        self.capture_memory()
        steady = self.steps[self.warmup_steps:]
        report = dict(
            scientific_result=False, scope='one full original-config epoch timing probe',
            stage=runner.cfg.original_runtime['stage'], rank=get_rank(), world_size=get_world_size(),
            completed_epochs=runner.epoch, train_iterations=len(self.steps),
            train_dataset_images=len(runner.train_dataloader.dataset),
            val_dataset_images=len(runner.val_dataloader.dataset),
            train_epoch_seconds=self.train_seconds, val_epoch_seconds=self.val_seconds,
            regular_checkpoint_seconds=self.regular_checkpoint_seconds,
            best_checkpoint_seconds=self.best_checkpoint_seconds,
            steady_epoch_seconds_estimate=self.steady_epoch_seconds,
            steady_compute_step_mean_seconds=statistics.mean(steady) if steady else None,
            warmup_steps_excluded_from_steady_estimate=self.warmup_steps,
            hook_elapsed_seconds=finished - self.started, parameters=self.parameter_counts,
            initialization_peak_allocated_bytes=self.initialization_peak,
            peak_allocated_bytes=self.peak_allocated, peak_reserved_bytes=self.peak_reserved,
            gpu=torch.cuda.get_device_name(), cuda_version=torch.version.cuda,
            device_total_bytes=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
            torch_version=torch.__version__, batch_per_gpu=runner.cfg.train_dataloader.batch_size,
            precision='fp32', cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark, loss_keys=self.loss_keys,
            original_validation_epochs=self.original_validation_epochs,
            metrics=getattr(self, 'metrics', None),
            checkpoint_save_calls=self.checkpoint_save_calls,
            timing_scope='epoch wall time includes loader/augmentation/model/optimizer/EMA; '
                         'validation includes metric aggregation; checkpoint gaps include '
                         'post-epoch hooks/logging and are reported separately',
            instrumentation='CUDA synchronized per training iteration; small timing overhead possible')
        path = Path(runner.work_dir) / f'runtime_rank_{get_rank()}.json'
        path.write_text(json.dumps(report, indent=2))


@HOOKS.register_module()
class OriginalEpochTimingTailHook(Hook):
    """Measure time after checkpoint hooks without replacing their implementation."""
    priority = 'LOWEST'

    def timing(self, runner):
        return next(h for h in runner.hooks if isinstance(h, OriginalEpochTimingHook))

    def after_train_epoch(self, runner):
        timing = self.timing(runner)
        if get_world_size() > 1:
            torch.distributed.barrier()
        timing.regular_checkpoint_seconds = timing.sync_time() - timing.train_ended
        timing.capture_memory()

    def after_val_epoch(self, runner, metrics=None):
        timing = self.timing(runner)
        if get_world_size() > 1:
            torch.distributed.barrier()
        timing.best_checkpoint_seconds = timing.sync_time() - timing.val_ended
        timing.capture_memory()
