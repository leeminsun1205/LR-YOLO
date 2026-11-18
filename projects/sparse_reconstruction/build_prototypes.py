"""Explicit one-pass clean TRAIN extraction. Run only via the S9 workflow."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmyolo.datasets.utils import yolov5_collate
from mmyolo.registry import DATASETS, MODELS
from mmyolo.utils import register_all_modules

from .prototype_bank import (POOLING, VERSION, QUALITY_POLICY, USABLE_STATUSES,
                             Reservoir, fit_bank, identity_hash, load_bank, role, sha256)
from .prototype_memory import pool_embeddings


def encoder_features(backbone, inputs):
    outputs = []
    x = inputs
    for index, name in enumerate(backbone.layers):
        x = getattr(backbone, name)(x)
        if index in backbone.out_indices:
            outputs.append(x)
    return tuple(outputs)


def collate_with_ids(items):
    # Preserve leading zeros even if an installed parser casts img_id to int.
    ids = [Path(item['data_samples'].img_path).stem for item in items]
    return ids, yolov5_collate(items)


def build(config):
    started = time.perf_counter()
    job = config.prototype_job
    destination = Path(job.directory)
    identity = dict(job.identity)
    expected = identity_hash(identity)
    if (destination / 'metadata.json').exists():
        _, metadata = load_bank(destination, expected)
        print('Reusing prototype cache:', destination, metadata['bank_sha256'], flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError('Incomplete prototype cache: preserve it and choose a new directory')
    if sha256(job.source) != identity['clean_source_sha256']:
        raise ValueError('Clean checkpoint changed before prototype extraction')
    saved = torch.load(job.source, map_location='cpu', weights_only=False)
    state = {k.removeprefix('module.'): v for k, v in saved['state_dict'].items()}
    backbone = MODELS.build(config.model.backbone)
    neck = MODELS.build(config.model.neck)
    preprocessor = MODELS.build(config.model.data_preprocessor)
    for prefix, module in [('backbone.', backbone), ('neck.', neck)]:
        module.init_weights()
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                if k.startswith(prefix)}, strict=True)
        # Native BaseBackbone/BaseYOLONeck.train() return None, so eval()
        # is not chainable even though it correctly sets evaluation mode.
        module.cuda()
        module.eval()
        module.requires_grad_(False)
    preprocessor.cuda().eval()
    del saved, state
    dataset = DATASETS.build(copy.deepcopy(config.train_dataloader.dataset))
    if tuple(dataset.metainfo['classes']) != tuple(identity['classes']):
        raise ValueError('Native parser class order differs from prototype contract')
    loader = torch.utils.data.DataLoader(dataset, batch_size=job.batch,
        shuffle=False, num_workers=job.workers, collate_fn=collate_with_ids,
        pin_memory=False, persistent_workers=job.workers > 0)
    classes = identity['classes']
    fit = [Reservoir(256, job.seed + 31*c) for c in range(len(classes))]
    audit = [Reservoir(64, job.seed + 31*c + 100003) for c in range(len(classes))]
    seen_ids, skipped = [], 0
    allowed = set(Path(config.train_dataloader.dataset.ann_file).read_text().split())
    feature_equality = False
    with torch.no_grad():
        for index, (ids, raw) in enumerate(loader):
            if any(image not in allowed for image in ids):
                raise ValueError('Prototype loader returned IDs outside TRAIN manifest')
            seen_ids.extend(ids)
            batch = preprocessor(raw, training=True)
            inputs = batch['inputs']
            if tuple(inputs.shape[-2:]) != (640, 640):
                raise ValueError('Prototype builder requires deterministic 640 letterboxing')
            if not feature_equality:
                # ONE small acceptance probe, not a second forward per stream batch.
                native, _ = backbone(inputs[:1])
                shortcut = encoder_features(backbone, inputs[:1])
                for a, b in zip(native, shortcut):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                native_neck = neck(native)
                shortcut_neck = neck(shortcut)
                for a, b in zip(native_neck, shortcut_neck):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                feature_equality = True
                del native, shortcut, native_neck, shortcut_neck
            features = encoder_features(backbone, inputs)
            p3 = neck(features)[0]
            if tuple(p3.shape[-2:]) != (80, 80):
                raise ValueError('Selected feature is not P3 stride8')
            packed = batch['data_samples']['bboxes_labels']
            # No cap for the offline reservoir stream: unbiased over eligible GT.
            rois = packed[:, [0, 2, 3, 4, 5]].clone()
            rois[:, [1, 3]] = rois[:, [1, 3]].clamp(0, 640)
            rois[:, [2, 4]] = rois[:, [2, 4]].clamp(0, 640)
            box_valid = torch.isfinite(rois).all(1) & (rois[:, 3] > rois[:, 1]) & (rois[:, 4] > rois[:, 2])
            vectors, vector_valid = pool_embeddings(p3, rois[box_valid])
            selected = packed[box_valid, :2].cpu().numpy()
            vectors = vectors.cpu().numpy()
            valid = vector_valid.cpu().numpy()
            skipped += int((~box_valid).sum()) + int((~valid).sum())
            for (image, label), vector, usable in zip(selected, vectors, valid):
                if label != int(label) or not 0 <= int(label) < len(classes):
                    raise ValueError('Native TRAIN GT class mismatch')
                if usable:
                    image_id = ids[int(image)]
                    group = audit if role(image_id, job.seed) == 'audit' else fit
                    group[int(label)].add(vector, image_id)
            if index % 100 == 0:
                print(f'Prototype extraction {index+1}/{len(loader)} batches', flush=True)
    if len(seen_ids) != len(set(seen_ids)) or not feature_equality:
        raise ValueError('Duplicate/empty prototype extraction stream')
    torch.cuda.synchronize()
    try:
        arrays, quality = fit_bank(fit, audit, classes)
    except ValueError as exc:
        arrays, quality = None, dict(status='rejected', failures=[str(exc)])
    metadata = dict(version=VERSION, identity=identity, identity_sha256=expected,
        pooling=POOLING, quality=quality, quality_policy=QUALITY_POLICY, source_feature_equality=feature_equality,
        clustering=dict(max_modes=2, iterations=20, minimum_cluster=8,
                        fit_capacity=256, audit_capacity=64, minimum_fit=32,
                        minimum_audit=16, minimum_audit_images=8,
                        initialization='class-mean nearest sample, then farthest sample'),
        source_state_key='state_dict', strict_loaded=['backbone', 'neck'],
        channels=None if arrays is None else arrays['prototypes'].shape[-1],
        processed_images=len(seen_ids), manifest_images=len(allowed),
        excluded_by_native_filter=sorted(allowed-set(seen_ids)), skipped_objects=skipped,
        fit_image_ids=sorted(i for i in seen_ids if role(i, job.seed) == 'fit'),
        audit_image_ids=sorted(i for i in seen_ids if role(i, job.seed) == 'audit'),
        reservoirs={name:dict(fit=[image for _,image in fit[c].items],
                              audit=[image for _,image in audit[c].items]) for c,name in enumerate(classes)},
        created_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started)
    for name in ('fit_image_ids', 'audit_image_ids'):
        metadata[name+'_sha256'] = identity_hash(metadata[name])
    if arrays is not None:
        np.savez(destination/'bank.npz', **arrays)
        metadata['bank_sha256'] = sha256(destination/'bank.npz')
    (destination/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if quality['status'] not in USABLE_STATUSES:
        raise ValueError('Clean prototype quality rejected: '+repr(quality['failures']))
    load_bank(destination, expected)
    print('Prototype bank:', quality['status'], destination, flush=True)
    print('Inactive auxiliary classes (native detection unchanged):',
          quality['inactive_auxiliary_classes'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    set_random_seed(cfg.prototype_job.seed)
    build(cfg)


if __name__ == '__main__':
    main()
