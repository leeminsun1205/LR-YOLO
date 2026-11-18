"""Additional S9 GPU acceptance checks; legacy S6/S8 runtime stays untouched."""
import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys

import torch
from mmengine.config import Config
from mmengine.runner import Runner, set_random_seed
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules

from . import run as legacy
from .build_prototypes import encoder_features
from .models import SelectiveReconstructionDetector
from .prototype_memory import (PrototypeMemorySelectiveDetector, bounded_rois,
                               margin_objective, pool_embeddings)


def close_tree(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b): raise AssertionError('Output tuple lengths changed')
        for x, y in zip(a, b): close_tree(x, y)
    elif isinstance(a, dict):
        if a.keys() != b.keys(): raise AssertionError('Loss/output keys changed')
        for key in a: close_tree(a[key], b[key])
    elif a is not None or b is not None:
        raise AssertionError('Unexpected output type')


def numerical_checks():
    """Small derivative/empty/border/multimode fixtures on the real CUDA ops."""
    import random
    import torch.nn.functional as F
    prototypes = torch.tensor([[[1.,0.],[0.,1.]],[[-1.,0.],[0.,0.]]], device='cuda')
    modes = torch.tensor([[True,True],[True,False]], device='cuda')
    margins = torch.full((2,), .1, device='cuda')
    roi = torch.tensor([[0.,0.,0.,1.,1.]], device='cuda')
    label = torch.zeros(1, dtype=torch.long, device='cuda')
    valid = torch.ones(1, dtype=torch.bool, device='cuda')
    second_mode = torch.tensor([[0.,1.]], device='cuda', requires_grad=True)
    satisfied, _ = margin_objective(second_mode, valid, roi, label, prototypes,modes,margins,1)
    torch.testing.assert_close(satisfied, satisfied.new_zeros(()), rtol=0, atol=0)
    raw = torch.tensor([[-1.,.2]], device='cuda', requires_grad=True)
    violated, _ = margin_objective(F.normalize(raw,dim=1),valid,roi,label,prototypes,modes,margins,1)
    grad = torch.autograd.grad(violated,raw)[0]
    assert violated > 0 and torch.isfinite(grad).all() and grad.abs().sum() > 0
    half, _ = margin_objective(F.normalize(raw,dim=1),valid,roi,label,prototypes,modes,margins,2)
    torch.testing.assert_close(half, violated/2)
    empty = raw[:0]
    zero, _ = margin_objective(empty,valid[:0],roi[:0],label[:0],prototypes,modes,margins,2)
    torch.testing.assert_close(zero, violated.new_zeros(()))
    assert torch.isfinite(torch.autograd.grad(zero,raw)[0]).all()
    feature = torch.zeros((1,2,80,80), device='cuda', requires_grad=True)
    borders = torch.tensor([[0.,0.,0.,.01,.01],[0.,639.,639.,640.,640.]], device='cuda')
    vectors, usable = pool_embeddings(feature,borders)
    assert not usable.any()
    zero, _ = margin_objective(vectors,usable,borders,label.expand(2),prototypes,modes,margins,1)
    assert zero == 0 and torch.isfinite(torch.autograd.grad(zero,feature)[0]).all()
    # Crop/ROI selection must leave native detection GT and global RNG intact.
    packed = torch.tensor([[0.,0.,i,i,i+1,i+1] for i in range(30)])
    original, state = packed.clone(), random.getstate()
    chosen, _ = bounded_rois(packed,1,640,640,2,19,42,0)
    assert len(chosen) == 8 and random.getstate() == state
    torch.testing.assert_close(packed,original,rtol=0,atol=0)


def audit_extra(cfg):
    numerical_checks()
    # Existing dense/crop/coverage/patch-objective/encoder checks remain intact.
    legacy.audit(cfg)
    model = MODELS.build(cfg.model)
    model.init_weights()
    saved = torch.load(cfg.load_from, map_location='cpu', weights_only=False)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in saved['state_dict'].items()}, strict=True)
    del saved
    model.cuda().train()
    if model.prototype_metadata.get('source_feature_equality') is not True:
        raise AssertionError('Cache lacks the native clean feature equality check')
    loader_cfg = copy.deepcopy(cfg.train_dataloader)
    loader_cfg.update(batch_size=1, num_workers=0, persistent_workers=False)
    loader = Runner.build_dataloader(loader_cfg, seed=cfg.randomness.seed)
    for raw in loader:
        batch = model.data_preprocessor(raw, training=True)
        packed = batch['data_samples']['bboxes_labels'].detach().cpu()
        rois, labels = bounded_rois(packed, 1, 640, 640, len(model.prototype_vectors), 0, cfg.randomness.seed, 0)
        if len(rois): break
    else:
        raise ValueError('S9 audit needs visible augmented GT')
    # No extra encoder is used in training. This audit-only probe isolates the
    # memory objective so reconstruction cannot mask a disconnected gradient.
    p3 = model.neck(encoder_features(model.backbone, batch['inputs']))[0]
    rois, labels = rois.cuda(), labels.cuda()
    vectors, valid = pool_embeddings(p3, rois)
    objective, _ = margin_objective(vectors, valid, rois, labels, model.prototype_vectors,
                                    model.prototype_modes, model.prototype_margins, 1)
    controlled = False
    if objective.detach().item() == 0:
        # A fully satisfied real batch legitimately has zero hinge. Force an
        # incorrect label ONLY for this derivative probe, never for training.
        similarity = torch.einsum('nd,ckd->nck', vectors.detach(), model.prototype_vectors)
        best = similarity.masked_fill(~model.prototype_modes[None], -torch.inf).max(2).values.argmax(1)
        labels = (best + 1) % len(model.prototype_vectors)
        objective, _ = margin_objective(vectors, valid, rois, labels, model.prototype_vectors,
                                        model.prototype_modes, model.prototype_margins, 1)
        controlled = True
    parameter = next(p for p in model.backbone.parameters() if p.requires_grad)
    gradient = torch.autograd.grad(objective, parameter)[0]
    if not torch.isfinite(gradient).all() or not gradient.abs().sum().item():
        raise AssertionError('Missing/nonfinite memory-only encoder gradient')
    if any(getattr(model, name).requires_grad for name in
           ('prototype_vectors', 'prototype_modes', 'prototype_margins')):
        raise AssertionError('Memory must be constant')
    del p3, vectors, valid, objective, gradient
    # SAME model/state/RNG, S8 loss versus S9 at lambda=0, one SGD update each.
    initial = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()

    def probe(memory):
        model.load_state_dict(initial, strict=True)
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1])
        model.prototype_weight = 0.
        losses = (model.loss if memory else
            lambda *args: SelectiveReconstructionDetector.loss(model, *args))(
                batch['inputs'], batch['gts'], batch['data_samples'])
        result = {k:v.detach().cpu().clone() for k,v in losses.items()}
        total, _ = model.parse_losses(losses)
        total.backward()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-5,
                                    momentum=.937, weight_decay=.0005, nesterov=True)
        optimizer.step()
        return result, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    a, after_a = probe(False)
    b, after_b = probe(True)
    close_tree(a, b); close_tree(after_a, after_b)
    del initial, after_a, after_b, a, b
    model.eval()
    assert PrototypeMemorySelectiveDetector.predict is SelectiveReconstructionDetector.predict
    with torch.no_grad():
        before = model.bbox_head(model.neck(encoder_features(model.backbone, batch['inputs'])))
        old = model.prototype_vectors.clone()
        model.prototype_vectors.zero_()
        after = model.bbox_head(model.neck(encoder_features(model.backbone, batch['inputs'])))
        model.prototype_vectors.copy_(old)
        close_tree(before, after)
    path = Path(cfg.work_dir)/'audit_report_rank_0.json'
    report = json.loads(path.read_text())
    report.update(prototype_memory_gradient=True, memory_gradient_controlled_labels=controlled,
        zero_weight_S8_losses_and_SGD_update_match=True, inference_independent_of_bank=True,
        cache_native_source_features_match=True, roi_margin_empty_border_fixtures_passed=True,
        prototype_checks_passed=False)
    path.write_text(json.dumps(report, indent=2)+'\n')


def lifecycle(cfg):
    """Tiny real DDP+EMA train/save/resume cycle, isolated from scientific runs."""
    gc.collect(); torch.cuda.empty_cache()
    root = Path(cfg.work_dir)/'technical_lifecycle'
    root.mkdir()
    gpus = min(2, torch.cuda.device_count())
    previous = None
    for end_epoch in (1, 2):
        probe = copy.deepcopy(cfg)
        folder = root/f'part{end_epoch}'
        folder.mkdir()
        probe.work_dir = str(folder)
        probe.launcher = 'pytorch' if gpus == 2 else 'none'
        probe.train_cfg.update(max_epochs=end_epoch, val_interval=1, dynamic_intervals=[])
        # Only the atomic resume slot is needed for this technical check.
        # Avoid duplicating full LPIPS/EMA tensors in regular/best checkpoints.
        probe.default_hooks.checkpoint.update(save_best=None, save_last=False,
                                             interval=100000, max_keep_ckpts=1)
        probe.custom_hooks = [h for h in probe.custom_hooks if h['type'] != 'BDDSessionBudgetHook']
        probe.sparse_contract.update(action='technical_lifecycle', target_epochs=2, world_size=gpus)
        if previous is not None:
            probe.load_from = str(previous)
            probe.resume = True
        path = folder/'cycle_config.py'
        probe.dump(str(path))
        command = [sys.executable]
        if gpus == 2:
            command += ['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
        command += ['tools/train.py',str(path)]
        if gpus == 2: command += ['--launcher','pytorch']
        log = folder/'cycle.log'
        try:
            with log.open('w') as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'S9 lifecycle failed; inspect {log}\n'+
                               '\n'.join(log.read_text(errors='replace').splitlines()[-80:])) from exc
        for rank in range(gpus):
            report = json.loads((folder/f'prototype_integrity_rank_{rank}.json').read_text())
            if (report['status'] != 'passed' or report['completed_epochs'] != end_epoch
                    or report['sampling_step'] != report['absolute_updates']-1):
                raise AssertionError('DDP/EMA/resume absolute-step or memory integrity failure')
        previous = folder/'latest_resume.pth'
        saved = torch.load(previous, map_location='cpu', weights_only=False)
        if not saved.get('optimizer', {}).get('state') or 'ema_state_dict' not in saved:
            raise AssertionError('Lifecycle checkpoint lacks optimizer/EMA history')
        if saved['meta']['epoch'] != end_epoch:
            raise AssertionError('Lifecycle did not resume the epoch loop')
        del saved
    path = Path(cfg.work_dir)/'audit_report_rank_0.json'
    report = json.loads(path.read_text())
    report.update(prototype_checks_passed=True, ddp_ema_save_resume_passed=True,
                  lifecycle_world_size=gpus, lifecycle_scope='technical 10-image / 2 epochs only',
                  lifecycle_checkpoint_bytes=sum(p.stat().st_size for p in root.rglob('*.pth')))
    path.write_text(json.dumps(report, indent=2)+'\n')
    print('S9 audit passed including save/resume:', path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--action', choices=['audit', 'profile'], required=True)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    set_random_seed(cfg.randomness.seed)
    if args.action == 'audit':
        audit_extra(cfg)
        lifecycle(cfg)
    else:
        # Profile the FULL auxiliary coefficient, not its near-zero early warmup.
        # This bool changes no weights, ROI cap, crop RNG or heavy forward count.
        cfg.model.prototype_profile = True
        legacy.profile(cfg, args.steps)
        from mmengine.dist import get_rank
        path = Path(cfg.work_dir)/f'profile_report_rank_{get_rank()}.json'
        report = json.loads(path.read_text())
        report['prototype_coefficient_profiled'] = cfg.model.prototype_weight
        report['bank_build_cost_included'] = False
        path.write_text(json.dumps(report, indent=2)+'\n')
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
