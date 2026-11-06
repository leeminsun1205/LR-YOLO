"""Run diagnostics without changing model losses or optimizer behavior."""
import json
import math
from pathlib import Path
import time

from mmengine.dist import get_rank, get_world_size
from mmengine.hooks import Hook
from mmyolo.registry import HOOKS
from .state import training_signature


def write_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


@HOOKS.register_module()
class BDDRunDiagnosticsHook(Hook):
    def __init__(self, record_step_times=False):
        self.record_step_times = record_step_times

    def before_train(self, runner):
        import torch
        self.started = time.monotonic()
        self.step_times = []
        torch.cuda.reset_peak_memory_stats()

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        if self.record_step_times:
            import torch
            torch.cuda.synchronize()
            self.step_started = time.monotonic()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for key, value in (outputs or {}).items():
            if 'loss' in key and not math.isfinite(float(value)):
                raise FloatingPointError(f'Non-finite {key} at iteration {runner.iter}')
        if self.record_step_times:
            import torch
            torch.cuda.synchronize()
            self.step_times.append(time.monotonic() - self.step_started)

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint.setdefault('meta', {})['bdd_run'] = training_signature(runner.cfg)

    def after_train(self, runner):
        import torch
        report = dict(elapsed_seconds=time.monotonic() - self.started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      device_total_bytes=torch.cuda.get_device_properties(
                          torch.cuda.current_device()).total_memory,
                      gpu_name=torch.cuda.get_device_name(),
                      rank=get_rank(), world_size=get_world_size(),
                      final_iteration=runner.iter)
        measured = self.step_times[2:]
        if measured:
            seconds = sum(measured) / len(measured)
            report.update(mean_step_seconds=seconds,
                          global_images_per_second=(
                              runner.cfg.train_dataloader.batch_size
                              * get_world_size() / seconds))
        write_atomic(Path(runner.work_dir) / f'compute_rank_{get_rank()}.json', report)
        if get_rank() == 0:
            write_atomic(Path(runner.work_dir) / 'compute.json', report)


@HOOKS.register_module()
class BDDResumePathsHook(Hook):
    """Rebase only file bookkeeping, keeping the previous best score intact."""
    priority = 'LOW'

    def before_train(self, runner):
        if not runner.cfg.resume:
            return
        hub = runner.message_hub
        old_best = hub.get_info('best_ckpt')
        if old_best:
            copied = Path(runner.work_dir) / Path(old_best).name
            if not copied.is_file():
                raise FileNotFoundError(f'Missing copied best checkpoint: {copied}')
            hub.update_info('best_ckpt', str(copied))
        # Epoch files in earlier sessions are never removed by retention here.
        hub.update_info('keep_ckpt_ids', [])


@HOOKS.register_module()
class BDDSessionBudgetHook(Hook):
    """Stop at a saved/validated epoch boundary without shortening the schedule."""
    priority = 'LOWEST'

    def __init__(self, deadline_unix=None, reserve_seconds=1800):
        self.deadline_unix = deadline_unix
        self.reserve_seconds = reserve_seconds
        self.longest_epoch = 0.

    def before_train_epoch(self, runner):
        self.epoch_started = time.monotonic()

    def after_val_epoch(self, runner, metrics=None):
        import torch
        self.longest_epoch = max(self.longest_epoch,
                                 time.monotonic() - self.epoch_started)
        completed = runner.epoch >= runner.max_epochs
        stop = (not completed and self.deadline_unix is not None
                and time.time() + self.longest_epoch * 1.25
                + self.reserve_seconds >= self.deadline_unix)
        flag = torch.tensor(int(stop if get_rank() == 0 else 0),
                            device=f'cuda:{torch.cuda.current_device()}')
        if get_world_size() > 1:
            torch.distributed.broadcast(flag, src=0)
        if flag.item():
            runner.train_loop.stop_training = True
        if get_rank() == 0:
            state = dict(status='completed' if completed else (
                'session_paused' if flag.item() else 'running'),
                completed_epochs=runner.epoch, target_epochs=runner.max_epochs,
                latest_checkpoint=runner.message_hub.get_info('last_ckpt'),
                best_checkpoint=runner.message_hub.get_info('best_ckpt'),
                longest_epoch_seconds=self.longest_epoch,
                deadline_unix=self.deadline_unix)
            write_atomic(Path(runner.work_dir) / 'session_state.json', state)
        if flag.item():
            runner.logger.info('Session budget reached; epoch checkpoint saved. '
                               'Resume in a new session with the original total epochs.')
