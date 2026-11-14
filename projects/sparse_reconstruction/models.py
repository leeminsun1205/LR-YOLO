"""Experimental late selective restoration; original detector files untouched."""
from collections import defaultdict

import torch
from torch import nn
from mmengine.dist import get_rank
from mmyolo.registry import MODELS
from mmyolo.models.detectors.yolo_reconstruct_dehaze_detector import YOLOReconstructDehazeDetector
from .sampling import sample_regions, decoder_bounds


@MODELS.register_module()
class SelectiveReconstructionDetector(YOLOReconstructDehazeDetector):
    def __init__(self, sparse_case=0, crop_count=2, crop_core=192,
                 crop_halo=32, sampling_seed=42, **kwargs):
        if sparse_case not in range(6):
            raise ValueError('Supported cases: S0..S5')
        super().__init__(**kwargs)
        self.sparse_case = sparse_case
        self.crop_count, self.crop_core, self.crop_halo = crop_count, crop_core, crop_halo
        self.sampling_seed, self.sampling_step = sampling_seed, 0
        self.last_sampling = {}

    def train(self, mode=True):
        super().train(mode)
        if self.sparse_case >= 1:
            # SAME state policy in controlled dense and patch cases. Encoder,
            # detection BN and original head training flags remain untouched.
            for name, module in self.backbone.named_children():
                if name.startswith(('up', 'skip', 'fusion', 'final_conv')):
                    for child in module.modules():
                        if isinstance(child, nn.modules.batchnorm._BatchNorm):
                            child.eval()
            self.criterion_lpips.eval()
        return self

    def encoder_and_trunk(self, inputs):
        b = self.backbone
        outs, features = [], []
        x = inputs
        for index, name in enumerate(b.layers):
            x = getattr(b, name)(x)
            features.append(x)
            if index in b.out_indices:
                outs.append(x)
        f1, f2, f3 = features[:3]
        z = b.up1(f3)
        z = b.up_conv1(b.fusion1([z, b.skip1(f2)]) + z)
        z = b.up2(z)
        z = b.up_conv2(b.fusion2([z, b.skip2(f1)]) + z)
        return tuple(outs), z

    def decode_regions(self, z, regions, height, width):
        # Group equal shapes: avoid launching one decoder per object. Keep
        # variable clipped halos at true image borders; never pad Z artificially.
        groups = defaultdict(list)
        for index, (image, region) in enumerate(regions):
            bounds = decoder_bounds(region, height, width)
            x0, y0, x1, y1, _, _ = bounds
            groups[(y1-y0, x1-x0)].append((index, image, region, bounds))
        result = [None] * len(regions)
        self.last_sampling['decoder_processed_area_ratio'] = sum(
            (shape[0]*2)*(shape[1]*2)*len(group) for shape,group in groups.items()
        )/(len(z)*height*width)
        for group in groups.values():
            crops = torch.stack([z[i, :, b[1]:b[3], b[0]:b[2]]
                                 for _, i, _, b in group])
            decoded = self.backbone.final_conv(self.backbone.up3(crops))
            for row, (index, _, region, bounds) in enumerate(group):
                dx, dy, size = bounds[4], bounds[5], region['size']
                result[index] = decoded[row, :, dy:dy+size, dx:dx+size]
        return torch.stack(result)

    def regions(self, inputs, samples):
        probability = {2: 0., 3: 0., 4: 1., 5: 0.5}[self.sparse_case]
        packed = samples['bboxes_labels'].detach().cpu() if isinstance(samples, dict) else None
        result = []
        for image in range(len(inputs)):
            boxes = (packed[packed[:, 0] == image, 2:6].tolist() if packed is not None
                     else samples[image].gt_instances.bboxes.detach().cpu().tolist())
            regions = sample_regions(*inputs.shape[-2:], boxes,
                count=self.crop_count, core=self.crop_core, halo=self.crop_halo,
                object_probability=probability,
                seed=self.sampling_seed + 1000003*self.sampling_step + 1009*get_rank() + image)
            result.extend((image, r) for r in regions)
        self.last_sampling = dict(regions=result, step=self.sampling_step,
                                  processed_area_ratio=sum(r['size']**2 for _, r in result)
                                  / (len(inputs)*inputs.shape[-1]*inputs.shape[-2]))
        return result

    def patch_losses(self, prediction, target, regions):
        l1 = []
        for index, (_, r) in enumerate(regions):
            x, y, side = r['core_x'], r['core_y'], r['core']
            l1.append(self.criterion_l1(prediction[index, :, y:y+side, x:x+side],
                                        target[index, :, y:y+side, x:x+side]))
        return dict(loss_reconstruct_l1=torch.stack(l1).mean()*self.loss_l1_weight,
                    # Mean across patches/images, independent of K. Keep source
                    # normalization; context-expanded patch is the LPIPS target.
                    loss_reconstruct_contrast=self.criterion_lpips(prediction, target).mean()
                    * self.loss_lpips_weight)

    def loss(self, batch_inputs, batch_gts, batch_data_samples):
        if batch_gts is None or batch_gts.shape != batch_inputs.shape:
            raise ValueError('Dark training needs aligned clean gts; never train with eval preprocessor')
        if self.sparse_case < 2:
            return super().loss(batch_inputs, batch_gts, batch_data_samples)
        features, z = self.encoder_and_trunk(batch_inputs)
        features = self.neck(features) if self.with_neck else features
        losses = self.bbox_head.loss(features, batch_data_samples)
        regions = self.regions(batch_inputs, batch_data_samples)
        if self.sparse_case == 2:
            dense = self.backbone.final_conv(self.backbone.up3(z))
            prediction = torch.stack([dense[i, :, r['y']:r['y']+r['size'],
                                                  r['x']:r['x']+r['size']] for i, r in regions])
        else:
            prediction = self.decode_regions(z, regions, *batch_inputs.shape[-2:])
        target = torch.stack([batch_gts[i, :, r['y']:r['y']+r['size'],
                                              r['x']:r['x']+r['size']] for i, r in regions])
        losses.update(self.patch_losses(prediction, target, regions))
        return losses
