# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import VOCDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class YOLOv5VOCFoggyDataset(BatchShapePolicyDataset, VOCDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with VOCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    METAINFO = {
        'classes': ('person', 'car', 'bus', 'bicycle',  'motorbike'), # 5 classes
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255)]
    }
