"""S9: S8 plus a fixed clean-source ROI cosine-margin objective, training only."""
import math
import random

import torch
import torch.nn.functional as F
from mmengine.dist import get_rank
from mmyolo.registry import MODELS
from torchvision.ops import roi_align

from .models import SelectiveReconstructionDetector
from .prototype_bank import POOLING, load_bank


def pool_embeddings(p3, rois):
    """Identical operator for clean-bank builder and degraded student."""
    pooled = roi_align(p3, rois.to(device=p3.device, dtype=p3.dtype),
        output_size=POOLING['output_size'], spatial_scale=1. / POOLING['stride'],
        sampling_ratio=POOLING['sampling_ratio'], aligned=POOLING['aligned']).mean((-2, -1))
    norms = pooled.norm(dim=1)
    valid = torch.isfinite(pooled).all(1) & (norms > POOLING['eps'])
    # nan_to_num protects the graph-connected zero path too.
    pooled = torch.nan_to_num(pooled)
    return F.normalize(pooled, dim=1, eps=POOLING['eps']), valid


def bounded_rois(packed_cpu, batch_size, height, width, classes, step, seed, rank, cap=8):
    """Private sampling; never remove native detection GT or consume global RNG."""
    if (packed_cpu.ndim != 2 or packed_cpu.shape[1] != 6
            or not torch.isfinite(packed_cpu).all()
            or ((packed_cpu[:,0] < 0) | (packed_cpu[:,0] >= batch_size)
                | (packed_cpu[:,0] != packed_cpu[:,0].round())).any()):
        raise ValueError('Invalid packed [batch, class, xyxy] GT')
    rows, labels = [], []
    for image in range(batch_size):
        eligible = []
        for row in packed_cpu[packed_cpu[:, 0] == image].tolist():
            label, x0, y0, x1, y1 = row[1:]
            if not all(math.isfinite(v) for v in row):
                raise ValueError('Nonfinite packed detection GT')
            if label != int(label) or not 0 <= int(label) < classes:
                raise ValueError('Prototype GT label outside native class order')
            x0, y0, x1, y1 = max(0., x0), max(0., y0), min(float(width), x1), min(float(height), y1)
            if x1 > x0 and y1 > y0:
                eligible.append((int(label), [image, x0, y0, x1, y1]))
        rng = random.Random(seed + 2000003 * step + 2017 * rank + image)
        chosen = rng.sample(eligible, cap) if len(eligible) > cap else eligible
        for label, roi in chosen:
            labels.append(label); rows.append(roi)
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5), torch.tensor(labels, dtype=torch.long)


def margin_objective(vectors, valid_vectors, rois, labels, prototypes, modes, margins, batch_size,
                     reliability=None):
    """Per-image mean then local-batch mean, including empty images as zero."""
    if not len(vectors):
        zero = vectors.sum() * 0.
        return zero, dict(prototype_active_fraction=zero.detach(), prototype_gap=zero.detach(),
                          prototype_supervised_fraction=zero.detach())
    if reliability is None:
        reliability = margins.new_ones(len(prototypes))
    similarity = torch.einsum('nd,ckd->nck', vectors, prototypes)
    similarity = similarity.masked_fill(~modes[None], -torch.inf).max(2).values
    positive = similarity.gather(1, labels[:, None]).squeeze(1)
    negative_mask = F.one_hot(labels, prototypes.shape[0]).bool() | (reliability <= 0)[None]
    negative = similarity.masked_fill(negative_mask, -torch.inf).max(1).values
    gap = positive - negative
    eligible = valid_vectors & (reliability[labels] > 0)
    terms = F.relu(margins[labels] - gap) * valid_vectors.to(gap.dtype) * reliability[labels]
    images = rois[:, 0].long()
    sums = terms.new_zeros(batch_size).scatter_add(0, images, terms)
    counts = terms.new_zeros(batch_size).scatter_add(0, images, valid_vectors.to(terms.dtype))
    loss = (sums / counts.clamp_min(1)).mean()
    denominator = valid_vectors.sum().clamp_min(1)
    return loss, dict(
        prototype_active_fraction=((terms > 0) & eligible).sum().detach() / denominator,
        prototype_supervised_fraction=eligible.sum().detach() / denominator,
        prototype_gap=(gap * eligible).sum().detach() / eligible.sum().clamp_min(1))


@MODELS.register_module()
class PrototypeMemorySelectiveDetector(SelectiveReconstructionDetector):
    def __init__(self, prototype_directory, prototype_identity, prototype_bank_sha256,
                 prototype_weight=1., prototype_warmup_updates=1, prototype_cap=8,
                 prototype_profile=False, **kwargs):
        if kwargs.get('sparse_case', 8) != 8:
            raise ValueError('S9 must preserve S8 sampling')
        if (not math.isfinite(prototype_weight) or not 0 <= prototype_weight <= 1
                or prototype_warmup_updates < 1 or prototype_cap != 8):
            raise ValueError('S9 uses weight [0,1], positive warmup and cap=8')
        kwargs['sparse_case'] = 8
        super().__init__(**kwargs)
        bank, metadata = load_bank(prototype_directory, prototype_identity)
        if metadata['bank_sha256'] != prototype_bank_sha256:
            raise ValueError('Bank content differs from the recorded training contract')
        for name, key in [('prototype_vectors', 'prototypes'), ('prototype_modes', 'valid'),
                          ('prototype_margins', 'margins'), ('prototype_reliability','reliability')]:
            self.register_buffer(name, torch.from_numpy(bank[key]), persistent=False)
        self.prototype_metadata = metadata
        self.prototype_weight = float(prototype_weight)
        self.prototype_warmup_updates = int(prototype_warmup_updates)
        self.prototype_cap = prototype_cap
        self.prototype_profile = bool(prototype_profile)

    def memory_loss(self, p3, packed_cpu, input_shape):
        height, width = input_shape
        if (p3.shape[1] != self.prototype_vectors.shape[-1]
                or tuple(p3.shape[-2:]) != (height // 8, width // 8)):
            raise ValueError('Prototype P3 channels/stride do not match clean extraction')
        rois, labels = bounded_rois(packed_cpu, len(p3), height, width,
            len(self.prototype_vectors), self.sampling_step, self.sampling_seed, get_rank(), self.prototype_cap)
        rois, labels = rois.to(p3.device), labels.to(p3.device)
        if not len(rois):
            zero = p3.sum() * 0.
            return zero, dict(prototype_active_fraction=zero.detach(), prototype_gap=zero.detach(),
                              prototype_supervised_fraction=zero.detach())
        vectors, valid = pool_embeddings(p3, rois)
        return margin_objective(vectors, valid, rois, labels, self.prototype_vectors,
            self.prototype_modes, self.prototype_margins, len(p3), self.prototype_reliability)

    def loss(self, batch_inputs, batch_gts, batch_data_samples):
        # Exact S8 branch at weight zero, including loss keys, RNG and updates.
        if self.prototype_weight == 0.:
            return super().loss(batch_inputs, batch_gts, batch_data_samples)
        if batch_gts is None or batch_gts.shape != batch_inputs.shape:
            raise ValueError('S9 training needs aligned clean reconstruction gts')
        features, z = self.encoder_and_trunk(batch_inputs)
        features = self.neck(features) if self.with_neck else features
        losses = self.bbox_head.loss(features, batch_data_samples)
        # S8 already transfers GT to CPU for crop selection. Reuse that transfer
        # for bounded ROI selection; pass only the unchanged packed GT to regions.
        packed = batch_data_samples['bboxes_labels'].detach().cpu()
        regions = self.regions(batch_inputs, dict(bboxes_labels=packed))
        prediction = self.decode_regions(z, regions, *batch_inputs.shape[-2:])
        target = torch.stack([batch_gts[i, :, r['y']:r['y']+r['size'],
                                               r['x']:r['x']+r['size']] for i, r in regions])
        losses.update(self.patch_losses(prediction, target, regions))
        memory, stats = self.memory_loss(features[0], packed, batch_inputs.shape[-2:])
        coefficient = self.prototype_weight * (1. if self.prototype_profile else
            min(1., self.sampling_step / self.prototype_warmup_updates))
        losses['loss_prototype_margin'] = memory * coefficient
        losses.update(stats)
        losses['prototype_coefficient'] = memory.detach().new_tensor(coefficient)
        return losses
