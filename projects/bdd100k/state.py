"""Portable experiment identity for epoch-checkpoint continuation."""
import hashlib
import json
from pathlib import Path


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_signature(cfg):
    """Ignore mount/work paths; preserve data, schedule, model and batch semantics."""
    signature = dict(
        model=cfg.model,
        optimizer=cfg.optim_wrapper,
        scheduler=cfg.default_hooks.param_scheduler,
        epochs=cfg.train_cfg.max_epochs,
        seed=cfg.randomness.seed,
        world_size=cfg.get('num_gpus', 1),
        batch_per_gpu=cfg.train_dataloader.batch_size,
        train_manifest=manifest_hash(cfg.train_dataloader.dataset.ann_file),
        val_manifest=manifest_hash(cfg.val_dataloader.dataset.ann_file),
        train_pipeline=cfg.train_dataloader.dataset.pipeline,
        val_pipeline=cfg.val_dataloader.dataset.pipeline,
        classes=cfg.train_dataloader.dataset.metainfo.classes)
    return json.loads(json.dumps(signature, sort_keys=True, default=str))
