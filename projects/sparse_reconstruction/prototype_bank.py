"""CPU bank fitting and integrity checks. No Torch import or global RNG changes."""
import hashlib
import json
import random
import copy
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LEGACY_VERSION = 'clean-prototype-margin-v1'
VERSION = 'clean-prototype-margin-v2-reliability'
USABLE_STATUSES = ('passed', 'usable_partial')
QUALITY_POLICY = dict(name='source_gap_reliability_v2', gap_scale=.1,
    minimum_reliable_classes=2, exclude_unreliable_negatives=True,
    minimum_fit=32, minimum_audit=16, minimum_audit_images=8)
# Known notebook78ea implementation; pooling/clustering are unchanged.
# Unknown or edited v1 implementations are NOT silently upgraded.
LEGACY_MODULE_HASHES = {
    'prototype_bank.py': '9cb5f3379e768a92cae78915d5aa00e7b7c779a5f665bfefcf5a0079c2cef578',
    'prototype_memory.py': '3687ecfdd5eebd3dd5f7fad9ef0a62745ccc36bc2b4bb28b4a8c1930adf1b804',
    'build_prototypes.py': '45a837f2437a3797d7772ab8a25354858a27dfce19a7c820bcdeba892e798478'}
POOLING = dict(level=0, stride=8, output_size=3, sampling_ratio=2, aligned=True,
               spatial_reduce='mean', normalization='l2', eps=1e-8)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def role(image_id, seed=42):
    value = hashlib.sha256(f'prototype-v1|{seed}|{image_id}'.encode()).hexdigest()
    return 'audit' if int(value, 16) % 5 == 0 else 'fit'


class Reservoir:
    def __init__(self, capacity, seed):
        self.capacity, self.seen = capacity, 0
        self.items = []
        self.rng = random.Random(seed)

    def add(self, vector, image_id):
        self.seen += 1
        item = (np.asarray(vector, dtype=np.float32).copy(), str(image_id))
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            index = self.rng.randrange(self.seen)
            if index < self.capacity:
                self.items[index] = item


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def fit_modes(vectors):
    """Deterministic spherical two-means, with an explicit single-mode fallback."""
    x = normalize(vectors)
    if x.ndim != 2 or len(x) < 32 or not np.isfinite(x).all():
        raise ValueError('Each class needs at least 32 finite fit vectors')
    if (np.linalg.norm(x, axis=1) < .99).any():
        raise ValueError('Zero/degenerate fit vectors')
    mean = x.mean(0)
    if np.linalg.norm(mean) < 1e-8:
        raise ValueError('Degenerate class mean; do not fabricate a prototype')
    first = int(np.argmax(x @ normalize(mean)))
    second = int(np.argmin(x @ x[first]))
    centers = x[[first, second]].copy()
    fallback = None
    for _ in range(20):
        assigned = (x @ centers.T).argmax(1)
        counts = np.bincount(assigned, minlength=2)
        if (counts < 8).any():
            fallback = 'cluster_support_below_8'
            break
        updated = np.stack([x[assigned == k].mean(0) for k in range(2)])
        if (np.linalg.norm(updated, axis=1) < 1e-8).any():
            fallback = 'degenerate_cluster'
            break
        centers = normalize(updated)
    assigned = (x @ centers.T).argmax(1)
    counts = np.bincount(assigned, minlength=2)
    if (counts < 8).any():
        fallback = fallback or 'final_cluster_support_below_8'
    if fallback:
        centers = normalize(mean)[None]
        counts = np.array([len(x)])
    result = np.zeros((2, x.shape[1]), np.float32)
    result[:len(centers)] = centers
    valid = np.arange(2) < len(centers)
    return result, valid, dict(fallback=fallback, counts=counts.tolist(),
        mean_cosine_distance=float((1 - (x @ centers.T).max(1)).mean()))


def fit_bank(fit, audit, classes):
    centers, masks, modes = [], [], []
    for name, reservoir in zip(classes, fit):
        if len(reservoir.items) < 32:
            raise ValueError(f'{name}: insufficient fit support ({len(reservoir.items)} < 32)')
        p, mask, detail = fit_modes(np.stack([v for v, _ in reservoir.items]))
        centers.append(p); masks.append(mask); modes.append(detail)
    centers, masks = np.stack(centers), np.stack(masks)
    diagnostics = {}
    for label, (name, reservoir) in enumerate(zip(classes, audit)):
        items = reservoir.items
        images = len({image for _, image in items})
        if not items:
            diagnostics[name] = dict(objects=0, images=images)
            continue
        x = np.stack([v for v, _ in items])
        similarity = np.einsum('nd,ckd->nck', x, centers)
        similarity[:, ~masks] = -np.inf
        scores = similarity.max(2)
        positive = scores[:, label].copy()
        accuracy = float((scores.argmax(1) == label).mean())
        scores[:, label] = -np.inf
        gaps = positive - scores.max(1)
        diagnostics[name] = dict(objects=len(items), images=images,
            accuracy=accuracy, gap_quantiles=np.quantile(gaps, [0,.25,.5,.75,1]).tolist(),
            positive_mode_usage=np.bincount(similarity[:, label].argmax(1), minlength=2).tolist(),
            fit_seen=fit[label].seen, audit_seen=reservoir.seen, modes=modes[label])
    flat = centers.reshape(-1, centers.shape[-1])
    margins, reliability, quality = calibrate_quality(diagnostics, classes)
    quality['mode_cosines'] = (flat @ flat.T).tolist()
    return dict(prototypes=centers, valid=masks, margins=margins, reliability=reliability), quality


def calibrate_quality(diagnostics, classes):
    """Separate invalid data from weak semantic priors; never fabricate a gap.

    v2 is a method revision: weak classes keep ALL native detection GT/losses,
    but supply neither prototype targets nor prototype negatives.
    """
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError('Prototype classes must contain at least two distinct labels')
    margins = np.zeros(len(classes), np.float32)
    reliability = np.zeros(len(classes), np.float32)
    failures, warnings, reliable, inactive = [], [], [], []
    for label, name in enumerate(classes):
        row = diagnostics.get(name, {})
        if (row.get('fit_seen', 0) < 32 or sum(row.get('modes', {}).get('counts', [])) < 32):
            failures.append(f'{name}: insufficient fit support')
        if row.get('objects', 0) < 16 or row.get('images', 0) < 8:
            failures.append(f'{name}: audit needs >=16 objects / >=8 images')
        gaps = np.asarray(row.get('gap_quantiles', []), dtype=np.float64)
        if gaps.shape != (5,) or not np.isfinite(gaps).all():
            failures.append(f'{name}: nonfinite/missing clean gap statistics')
            inactive.append(name)
            continue
        median = float(gaps[2])
        if median <= 0:
            warnings.append(f'{name}: median clean gap {median}; auxiliary prototype weight=0')
            inactive.append(name)
        else:
            margins[label] = min(.1, .5 * median)
            reliability[label] = min(1., median / QUALITY_POLICY['gap_scale'])
            if margins[label] > 0 and reliability[label] > 0:
                reliable.append(name)
            else:
                margins[label] = reliability[label] = 0
                inactive.append(name)
                failures.append(f'{name}: positive gap is not representable in FP32')
    if len(reliable) < QUALITY_POLICY['minimum_reliable_classes']:
        failures.append('Need at least two reliable classes for an inter-class prototype objective')
    status = 'rejected' if failures else ('usable_partial' if inactive else 'passed')
    return margins, reliability, dict(status=status, failures=failures, warnings=warnings,
        separation_status='partial' if inactive else 'complete', classes=copy.deepcopy(diagnostics),
        reliable_classes=reliable, inactive_auxiliary_classes=inactive,
        reliability={name:float(reliability[i]) for i,name in enumerate(classes)},
        balanced_accuracy=float(np.mean([diagnostics.get(n,{}).get('accuracy',0.) for n in classes])),
        policy=QUALITY_POLICY)


def load_bank(directory, expected_identity=None):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('version') != VERSION or metadata.get('quality', {}).get('status') not in USABLE_STATUSES:
        raise ValueError('Prototype cache version/quality failed; inspect metadata.json')
    identity = metadata['identity']
    if metadata.get('identity_sha256') != identity_hash(identity):
        raise ValueError('Prototype metadata identity hash mismatch')
    if expected_identity is not None and metadata['identity_sha256'] != expected_identity:
        raise ValueError('Prototype cache identity differs from the training contract')
    if sha256(directory / 'bank.npz') != metadata['bank_sha256']:
        raise ValueError('Prototype bank hash mismatch')
    with np.load(directory / 'bank.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ('prototypes', 'valid', 'margins', 'reliability')}
    p, valid, margin = arrays['prototypes'], arrays['valid'], arrays['margins']
    classes = identity['classes']
    weight = arrays['reliability']
    if (p.ndim != 3 or p.shape[:2] != (len(classes), 2) or p.dtype != np.float32
            or valid.shape != p.shape[:2] or valid.dtype != np.bool_
            or margin.shape != (len(classes),) or margin.dtype != np.float32
            or not valid.any(1).all() or not np.isfinite(p).all()
            or not np.isfinite(margin).all() or (margin < 0).any() or (margin > .100001).any()
            or weight.shape != (len(classes),) or weight.dtype != np.float32
            or not np.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any()
            or (weight > 0).sum() < 2 or not np.array_equal(weight > 0, margin > 0)
            or not np.allclose(np.linalg.norm(p[valid], axis=1), 1., atol=1e-5)):
        raise ValueError('Invalid prototype shapes, modes, margins or normalization')
    expected_margin, expected_weight, expected_quality = calibrate_quality(metadata['quality']['classes'], classes)
    if (any(metadata['quality'].get(key) != expected_quality[key] for key in
            ('policy','status','reliable_classes','inactive_auxiliary_classes','separation_status','reliability'))
            or expected_quality['status'] not in USABLE_STATUSES
            or not np.array_equal(expected_margin, margin) or not np.array_equal(expected_weight, weight)):
        raise ValueError('Prototype reliability/margins do not match measured source diagnostics')
    return arrays, metadata


def legacy_identity_matches(identity, expected):
    ignored = {'version', 'module_sha256', 'quality_policy'}
    return (identity.get('version') == LEGACY_VERSION
        and identity.get('module_sha256') == LEGACY_MODULE_HASHES
        and identity_hash({k:v for k,v in identity.items() if k not in ignored}) ==
            identity_hash({k:v for k,v in expected.items() if k not in ignored}))


def upgrade_legacy_cache(source, destination, identity):
    """Reuse known v1 source features; write a NEW v2 artifact, never edit v1."""
    source, destination = Path(source), Path(destination)
    started = time.perf_counter()
    metadata = json.loads((source/'metadata.json').read_text())
    if (metadata.get('version') != LEGACY_VERSION
            or not legacy_identity_matches(metadata.get('identity',{}), identity)
            or metadata.get('identity_sha256') != identity_hash(metadata['identity'])
            or metadata.get('source_feature_equality') is not True
            or metadata.get('bank_sha256') != sha256(source/'bank.npz')):
        raise ValueError('Cannot reuse unverified or incompatible v1 prototype cache')
    with np.load(source/'bank.npz', allow_pickle=False) as archive:
        arrays = {key:archive[key].copy() for key in ('prototypes','valid','margins')}
    old_quality = metadata['quality']
    legacy_bank_sha = metadata['bank_sha256']
    margins, weights, quality = calibrate_quality(old_quality.get('classes',{}), identity['classes'])
    if quality['status'] not in USABLE_STATUSES:
        raise ValueError('Legacy cache is structurally unusable: '+repr(quality['failures']))
    quality['mode_cosines'] = old_quality.get('mode_cosines', [])
    # Validate inherited prototype tensor geometry BEFORE writing the new cache.
    p, valid = arrays['prototypes'], arrays['valid']
    if (p.ndim != 3 or p.shape[:2] != (len(identity['classes']),2) or p.dtype != np.float32
            or valid.shape != p.shape[:2] or valid.dtype != np.bool_ or not valid.any(1).all()
            or not np.isfinite(p).all() or not np.allclose(np.linalg.norm(p[valid],axis=1),1.,atol=1e-5)):
        raise ValueError('Legacy prototype vectors/modes are invalid')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Preserve existing destination cache: '+str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    arrays.update(margins=margins, reliability=weights)
    np.savez(destination/'bank.npz', **arrays)
    metadata = copy.deepcopy(metadata)
    metadata.update(version=VERSION, identity=identity, identity_sha256=identity_hash(identity),
        quality=quality, quality_policy=QUALITY_POLICY, bank_sha256=sha256(destination/'bank.npz'),
        calibration_created_utc=datetime.now(timezone.utc).isoformat(),
        calibration_elapsed_seconds=time.perf_counter()-started,
        migration=dict(source_directory=str(source), source_metadata_sha256=sha256(source/'metadata.json'),
            source_bank_sha256=legacy_bank_sha,
            original_version=LEGACY_VERSION, original_quality=old_quality, cached_extraction_reused=True))
    (destination/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    load_bank(destination, identity_hash(identity))
    return metadata
