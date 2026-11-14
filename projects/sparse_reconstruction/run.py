"""Explicit GPU audit/profile/final evaluation; never runs on import."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch
from mmengine.config import Config
from mmengine.dist import get_rank, get_world_size
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner, set_random_seed
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules
from .hooks import frozen_hash


def write(cfg, action, report):
    path = Path(cfg.work_dir)/f'{action}_report_rank_{get_rank()}.json'
    path.write_text(json.dumps(report, indent=2))
    print('report:', path, flush=True)


def audit(cfg):
    if cfg.sparse_contract['stage'] != 'dark':
        raise ValueError('Audit is dark-only; use clean smoke first')
    model = MODELS.build(cfg.model)
    model.init_weights()
    saved = torch.load(cfg.load_from, map_location='cpu', weights_only=False)
    model.load_state_dict({k.removeprefix('module.'): v for k,v in saved['state_dict'].items()}, strict=True)
    model.cuda().eval()
    loader_cfg = copy.deepcopy(cfg.train_dataloader)
    loader_cfg.batch_size = 1
    loader_cfg.num_workers = 0
    loader_cfg.persistent_workers = False
    raw = next(iter(Runner.build_dataloader(loader_cfg, seed=cfg.randomness.seed)))
    batch = model.data_preprocessor(raw, training=True)
    with torch.no_grad():
        original_features, _ = model.backbone(batch['inputs'])
        new_features, z = model.encoder_and_trunk(batch['inputs'])
        for a,b in zip(original_features, new_features):
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    # Test actual late convolutions, PixelShuffle and RGB BN at both real
    # borders, an interior location and overlapping crops. No LPIPS shortcut.
    side = model.crop_core + 2*model.crop_halo
    height,width = batch['inputs'].shape[-2:]
    regions = [(0, dict(x=x,y=y,size=side)) for x,y in
               [(0,0),(width-side,height-side),(min(16,width-side),min(32,height-side)),
                (min(32,width-side),min(32,height-side))]]
    z = z.detach().requires_grad_(True)
    dense = model.backbone.final_conv(model.backbone.up3(z))
    reference = torch.stack([dense[i,:,r['y']:r['y']+side,r['x']:r['x']+side]
                             for i,r in regions])
    sparse = model.decode_regions(z, regions, height,width)
    torch.testing.assert_close(reference,sparse,rtol=2e-4,atol=2e-5)
    first = torch.autograd.grad(reference.square().mean(), z, retain_graph=True)[0]
    second = torch.autograd.grad(sparse.square().mean(), z)[0]
    torch.testing.assert_close(first,second,rtol=5e-4,atol=1e-7)
    del dense,reference,sparse,first,second,z
    model.train()
    if model.sparse_case >= 1:
        assert not model.criterion_lpips.training
        restoration_bn = [child for name,module in model.backbone.named_children()
                          if name.startswith(('up','skip','fusion','final_conv'))
                          for child in module.modules()
                          if isinstance(child,torch.nn.modules.batchnorm._BatchNorm)]
        if any(child.training for child in restoration_bn):
            raise AssertionError('Controlled restoration BN entered training mode')
    if not model.bbox_head.head_module.training:
        raise AssertionError('Head must retain training output contract for DFL loss')
    before = frozen_hash(model)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    losses = model.loss(batch['inputs'],batch['gts'],batch['data_samples'])
    total,_ = model.parse_losses(losses)
    reconstruction = losses['loss_reconstruct_l1'] + losses['loss_reconstruct_contrast'].mean()
    encoder_parameter = next(p for n,p in model.backbone.named_parameters() if p.requires_grad)
    reconstruction_gradient = torch.autograd.grad(reconstruction, encoder_parameter, retain_graph=True)[0]
    if not torch.isfinite(reconstruction_gradient).all() or not reconstruction_gradient.abs().sum().item():
        raise AssertionError('Missing/nonfinite reconstruction gradient into encoder')
    previous = encoder_parameter.detach().clone()
    total.backward()
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise AssertionError('Frozen parameters received gradients')
    if not torch.isfinite(total) or any(not torch.isfinite(p.grad).all() for p in model.parameters()
                                      if p.grad is not None):
        raise FloatingPointError('Nonfinite audit gradient/loss')
    optimizer.step()
    if torch.equal(previous,encoder_parameter) or frozen_hash(model) != before:
        raise AssertionError('Encoder did not update or frozen weights changed')
    write(cfg,'audit',dict(status='passed',scientific_result=False,
        source_detection_features_match=True, late_crop_RGB_and_input_gradients_match=True,
        finite_encoder_reconstruction_gradient=True, encoder_updated=True,
        frozen_weights_unchanged=True, loss=float(total.detach()),
        sampling=model.last_sampling,contract=dict(cfg.sparse_contract),
        source_sha256=cfg.sparse_initialization['sha256']))


def profile(cfg, steps):
    if steps < 10:
        raise ValueError('Measure at least 10 steps after warmup')
    runner = Runner.from_cfg(cfg)
    runner.optim_wrapper = runner.build_optim_wrapper(cfg.optim_wrapper)
    model = runner.model
    # Follow pinned Runner.train initialization/load order without executing
    # the epoch loop; source tensor contracts are checked in before_run.
    runner.call_hook('before_run')
    runner._init_model_weights()
    runner.load_or_resume()
    model.train()
    loader = runner.train_dataloader
    iterator = iter(loader)
    training_hooks = [h for h in runner.hooks if type(h).__name__ in
                      ('YOLOv5ParamSchedulerHook', 'EMAHook')]
    if {type(h).__name__ for h in training_hooks} != {'YOLOv5ParamSchedulerHook', 'EMAHook'}:
        raise RuntimeError('Profile requires the same original scheduler and EMA as training')
    for hook in training_hooks:
        hook.before_train(runner)
    warmup = 10
    runner.optim_wrapper.initialize_count_status(model, init_counts=0, max_counts=warmup+steps)
    measured_start = None
    peak_a = peak_r = 0
    for step in range(warmup+steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if step == warmup:
            torch.cuda.synchronize()
            if get_world_size() > 1:
                torch.distributed.barrier()
            torch.cuda.reset_peak_memory_stats()
            peak_a = peak_r = 0
            measured_start = time.perf_counter()
        bare = model.module if is_model_wrapper(model) else model
        runner.train_loop._iter = step
        bare.sampling_step = step
        for hook in training_hooks:
            hook.before_train_iter(runner, step, data_batch=batch)
        output = model.train_step(batch, runner.optim_wrapper)
        for hook in training_hooks:
            hook.after_train_iter(runner, step, data_batch=batch, outputs=output)
        runner.train_loop._iter = step+1
        if any(not torch.isfinite(torch.as_tensor(v)).all() for k,v in output.items() if 'loss' in k):
            raise FloatingPointError('Nonfinite profile loss')
        peak_a = max(peak_a,torch.cuda.max_memory_allocated())
        peak_r = max(peak_r,torch.cuda.max_memory_reserved())
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-measured_start
    write(cfg,'profile',dict(scientific_result=False,steps=steps,warmup_steps=warmup,
        elapsed_seconds=elapsed,mean_step_seconds=elapsed/steps,
        global_images_per_second=steps*cfg.train_dataloader.batch_size*get_world_size()/elapsed,
        peak_allocated_bytes=peak_a,peak_reserved_bytes=peak_r,world_size=get_world_size(),
        case_id=cfg.sparse_contract['case_id'], gpu=torch.cuda.get_device_name(),
        last_sampling=getattr(bare,'last_sampling',{}),
        scope='training steps incl SGD/clipping/DDP/original warmup scheduler/EMA; '
              'excludes validation/checkpoints/logger; use pilot epoch for complete cost'))


def evaluate(cfg):
    from projects.lowlight_adapter.metrics import VOCIncludingDifficultMetric  # noqa
    data = cfg.sparse_evaluation
    for key in ('train_dataloader','train_cfg','val_dataloader','val_cfg','val_evaluator',
                'optim_wrapper','param_scheduler'):
        cfg.pop(key,None)
    cfg.custom_hooks = []
    cfg.test_evaluator = [dict(type='mmdet.VOCMetric', metric='mAP', eval_mode='11points',
        iou_thrs=0.5,prefix='voc_standard'),dict(type='VOCIncludingDifficultMetric',
        metric='mAP',eval_mode='11points',iou_thrs=0.5,prefix='voc_all_objects')]
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    write(cfg,'evaluate',dict(metrics=metrics,images=len(runner.test_dataloader.dataset),
        evaluation=data,precision='fp32',scope='final evaluation; unchanged full decoder prediction',
        paper_all_objects_equivalence='UNKNOWN; source legacy coordinate/ignore policies disclosed'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--action',choices=['audit','profile','evaluate'],required=True)
    parser.add_argument('--steps',type=int,default=100)
    args=parser.parse_args()
    register_all_modules()
    cfg=Config.fromfile(args.config)
    set_random_seed(cfg.randomness.seed)
    if args.action == 'audit':
        audit(cfg)
    elif args.action == 'profile':
        profile(cfg,args.steps)
    else:
        evaluate(cfg)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
