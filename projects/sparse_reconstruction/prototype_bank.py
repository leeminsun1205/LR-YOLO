"""CPU bank fitting and integrity checks. No Torch import or global RNG changes."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

VERSION = 'clean-prototype-margin-v1'
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
    diagnostics, margins, failures = {}, [], []
    for label, (name, reservoir) in enumerate(zip(classes, audit)):
        items = reservoir.items
        images = len({image for _, image in items})
        if len(items) < 16 or images < 8:
            failures.append(f'{name}: audit needs >=16 objects / >=8 images; got {len(items)} / {images}')
        if not items:
            diagnostics[name] = dict(objects=0, images=images)
            margins.append(0.)
            continue
        x = np.stack([v for v, _ in items])
        similarity = np.einsum('nd,ckd->nck', x, centers)
        similarity[:, ~masks] = -np.inf
        scores = similarity.max(2)
        positive = scores[:, label].copy()
        accuracy = float((scores.argmax(1) == label).mean())
        scores[:, label] = -np.inf
        gaps = positive - scores.max(1)
        median = float(np.median(gaps))
        if median <= 0 or not np.isfinite(median):
            failures.append(f'{name}: non-positive/nonfinite median clean gap ({median})')
        margins.append(max(0., min(.1, .5 * median)))
        diagnostics[name] = dict(objects=len(items), images=images,
            accuracy=accuracy, gap_quantiles=np.quantile(gaps, [0,.25,.5,.75,1]).tolist(),
            positive_mode_usage=np.bincount(similarity[:, label].argmax(1), minlength=2).tolist(),
            fit_seen=fit[label].seen, audit_seen=reservoir.seen, modes=modes[label])
    flat = centers.reshape(-1, centers.shape[-1])
    return dict(prototypes=centers, valid=masks, margins=np.asarray(margins, np.float32)), dict(
        status='rejected' if failures else 'passed', failures=failures, classes=diagnostics,
        balanced_accuracy=float(np.mean([d.get('accuracy', 0.) for d in diagnostics.values()])),
        mode_cosines=(flat @ flat.T).tolist())


def load_bank(directory, expected_identity=None):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('version') != VERSION or metadata.get('quality', {}).get('status') != 'passed':
        raise ValueError('Prototype cache version/quality failed; inspect metadata.json')
    identity = metadata['identity']
    if metadata.get('identity_sha256') != identity_hash(identity):
        raise ValueError('Prototype metadata identity hash mismatch')
    if expected_identity is not None and metadata['identity_sha256'] != expected_identity:
        raise ValueError('Prototype cache identity differs from the training contract')
    if sha256(directory / 'bank.npz') != metadata['bank_sha256']:
        raise ValueError('Prototype bank hash mismatch')
    with np.load(directory / 'bank.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ('prototypes', 'valid', 'margins')}
    p, valid, margin = arrays['prototypes'], arrays['valid'], arrays['margins']
    classes = identity['classes']
    if (p.ndim != 3 or p.shape[:2] != (len(classes), 2) or p.dtype != np.float32
            or valid.shape != p.shape[:2] or valid.dtype != np.bool_
            or margin.shape != (len(classes),) or margin.dtype != np.float32
            or not valid.any(1).all() or not np.isfinite(p).all()
            or not np.isfinite(margin).all() or (margin <= 0).any() or (margin > .100001).any()
            or not np.allclose(np.linalg.norm(p[valid], axis=1), 1., atol=1e-5)):
        raise ValueError('Invalid prototype shapes, modes, margins or normalization')
    return arrays, metadata
