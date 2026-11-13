"""Explicit single-GPU smoke/evaluation entry point; never trains on test data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from mmengine.config import Config
from mmengine.runner import Runner, set_random_seed
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from .hooks import tensor_fingerprint


def write_report(cfg, action, report):
    folder = Path(cfg.work_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f'{action}_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def load_model(cfg, checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = MODELS.build(cfg.model)
    model.init_weights()
    # Notebook writes a normalized, fully validated complete state beforehand.
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.cuda().eval()


def tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [t for key in sorted(value) for t in tensors(value[key])]
    if isinstance(value, (tuple, list)):
        return [t for child in value for t in tensors(child)]
    if value is None:
        return []
    raise TypeError(type(value))


def check_equivalence(model, batch):
    with torch.no_grad():
        original = model.frozen_features(batch['inputs'], decoder_free=False)
        fast = model.frozen_features(batch['inputs'], decoder_free=True)
        if len(original) != len(fast):
            raise AssertionError('Feature count differs')
        for first, second in zip(original, fast):
            torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
        original_outputs = tensors(model.bbox_head.forward(original))
        fast_outputs = tensors(model.bbox_head.forward(fast))
        if len(original_outputs) != len(fast_outputs):
            raise AssertionError('Head output count differs')
        for first, second in zip(original_outputs, fast_outputs):
            torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    return 'full decoder vs bypass: neck features and raw head outputs close'


def smoke(cfg, checkpoint):
    model = load_model(cfg, checkpoint)
    loader_cfg = copy.deepcopy(cfg.val_dataloader)
    loader_cfg.num_workers = 0
    loader_cfg.persistent_workers = False
    loader = Runner.build_dataloader(loader_cfg, seed=cfg.randomness.seed)
    batch = model.data_preprocessor(next(iter(loader)), training=False)
    equivalence = check_equivalence(model, batch)
    report = dict(scope='technical_smoke_only', scientific_result=False,
                  case_id=cfg.adapter_contract['case_id'], equivalence=equivalence,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if report['trainable_parameters']:
        train_loader_cfg = copy.deepcopy(cfg.train_dataloader)
        train_loader_cfg.num_workers = 0
        train_loader_cfg.persistent_workers = False
        loader = Runner.build_dataloader(train_loader_cfg, seed=cfg.randomness.seed)
        model.train()
        batch = model.data_preprocessor(next(iter(loader)), training=True)
        # Identity test is meaningful only for freshly initialized adapters.
        with torch.no_grad():
            features = model.frozen_features(batch['inputs'])
            adapted = model.extract_feat(batch['inputs'])
            identity = all(torch.allclose(a, b, rtol=1e-5, atol=1e-6)
                           for a, b in zip(features, adapted))
        if cfg.get('adapter_identity_expected', False) and not identity:
            raise AssertionError('Fresh adapters must preserve initial features')
        before = tensor_fingerprint(model.state_dict(), False)
        previous = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.001)
        losses = model.loss(batch['inputs'], batch['data_samples'])
        loss, _ = model.parse_losses(losses)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite smoke detection loss')
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise AssertionError('Adapter gradients missing/non-finite')
        if not any(torch.count_nonzero(g).item() for g in gradients):
            raise AssertionError('All adapter gradients are zero')
        if any(p.grad is not None for n, p in model.named_parameters() if not p.requires_grad):
            raise AssertionError('Frozen parameters received gradients')
        optimizer.step()
        if not any(not torch.equal(previous[n], p) for n, p in model.named_parameters()
                   if p.requires_grad):
            raise AssertionError('Adapter weights did not update')
        if tensor_fingerprint(model.state_dict(), False) != before:
            raise AssertionError('Frozen parameters/buffers changed in smoke')
        report.update(identity_at_start=identity, loss=float(loss.detach()),
                      finite_nonzero_adapter_gradient=True, frozen_state_unchanged=True)
    write_report(cfg, 'smoke', report)


def benchmark(model, cfg, image_count):
    loader = Runner.build_dataloader(cfg.test_dataloader, seed=cfg.randomness.seed)
    elapsed = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, raw in enumerate(loader):
            if index >= image_count:
                break
            batch = model.data_preprocessor(raw, training=False)
            if index == 0:
                for _ in range(5):
                    model.predict(batch['inputs'], batch['data_samples'])
                torch.cuda.synchronize()
            torch.cuda.synchronize()
            started = time.perf_counter()
            model.predict(batch['inputs'], batch['data_samples'])
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)
    if not elapsed:
        raise RuntimeError('Empty benchmark loader')
    return dict(images=len(elapsed), batch_size=1, mean_ms=1000 * statistics.mean(elapsed),
                median_ms=1000 * statistics.median(elapsed),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                cuda_version=torch.version.cuda, precision='fp32', warmup_forwards=5,
                scope='model prediction incl. postprocessing; excludes loading/preprocessing/visualization',
                decoder_parameters_retained_in_memory=True)


def evaluate(cfg, checkpoint, image_count):
    eval_cfg = copy.deepcopy(cfg)
    for key in ('train_dataloader', 'train_cfg', 'val_dataloader', 'val_cfg',
                'val_evaluator', 'optim_wrapper', 'param_scheduler'):
        eval_cfg.pop(key, None)
    eval_cfg.custom_hooks = []
    eval_cfg.load_from = str(checkpoint)
    runner = Runner.from_cfg(eval_cfg)
    metrics = runner.test()
    model = runner.model
    fingerprint = tensor_fingerprint(model.state_dict())
    latency = benchmark(model, eval_cfg, image_count) if image_count else None
    sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    source_contract = cfg.get('adapter_source_training_contract')
    pilot = source_contract is not None and source_contract.get('train_count') != 12334
    count = len(runner.test_dataloader.dataset)
    expected = 3760 if cfg.adapter_evaluation_domain == 'voc' else 2563
    report = dict(scope='full_evaluation' if count == expected and not pilot else 'pilot_evaluation',
                  case_id=cfg.adapter_contract['case_id'],
                  domain=cfg.adapter_evaluation_domain, metrics=metrics,
                  checkpoint_sha256=sha, model_sha256=fingerprint,
                  base_checkpoint_sha256=cfg.adapter_contract['base_checkpoint_sha256'],
                  evaluation_manifest_sha256=cfg.adapter_contract['val_manifest_sha256'],
                  images=count, source_training_contract=source_contract,
                  source_checkpoint_epoch=cfg.get('adapter_source_epoch'),
                  parameters=dict(resident_parameters=sum(p.numel() for p in model.parameters()),
                                  adapter_parameters=sum(p.numel() for p in model.adapters.parameters()),
                                  decoder_executed=not model.decoder_free),
                  metric='VOC AP50, 11-point, legacy coordinates',
                  all_objects_policy='ignored/difficult boxes promoted to ordinary GT',
                  paper_all_objects_equivalence='UNKNOWN; author evaluator not independently verified',
                  selected_before_test=cfg.get('adapter_selection_locked', False),
                  latency=latency)
    write_report(cfg, 'evaluate', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--action', choices=('smoke', 'evaluate'), required=True)
    parser.add_argument('--benchmark-images', type=int, default=16)
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    set_random_seed(cfg.randomness.seed)
    if args.action == 'smoke':
        smoke(cfg, args.checkpoint)
    else:
        evaluate(cfg, args.checkpoint, args.benchmark_images)


if __name__ == '__main__':
    main()
