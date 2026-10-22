# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import VOCDataset

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset
from ..registry import DATASETS


@DATASETS.register_module()
class YOLOv5VOCDarkDataset(BatchShapePolicyDataset, VOCDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with VOCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    METAINFO = {
        'classes': ('person', 'car', 'bus', 'bicycle',  'motorbike', 
                    'boat', 'bottle', 'chair', 'dog', 'cat'), # 10 classes
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255),
                    (0, 60, 100), (0, 0, 142), (255, 77, 255), (153, 69, 1), (120, 166, 157)]
    }


@DATASETS.register_module()
class YOLOv5ExDarkDataset(BatchShapePolicyDataset, VOCDataset):
    """Dataset for YOLOv5 VOC Dataset.

    We only add `BatchShapePolicy` function compared with VOCDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    METAINFO = {
        'classes': ('People', 'Car', 'Bus', 'Bicycle',  'Motorbike', 
                    'Boat', 'Bottle', 'Chair', 'Dog', 'Cat'), # 10 classes
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255),
                    (0, 60, 100), (0, 0, 142), (255, 77, 255), (153, 69, 1), (120, 166, 157)]
    }
    
@DATASETS.register_module()
class YOLOv5BDD100KNightDataset(BatchShapePolicyDataset, VOCDataset):
    """BDD100K-Night converted to Pascal VOC format."""

    METAINFO = {
        'classes': (
            'pedestrian',
            'rider',
            'car',
            'truck',
            'bus',
            'train',
            'motorcycle',
            'bicycle',
            'traffic light',
            'traffic sign'
        ),
        'palette': [
            (106, 0, 228),
            (119, 11, 32),
            (165, 42, 42),
            (0, 0, 192),
            (197, 226, 255),
            (0, 60, 100),
            (0, 0, 142),
            (255, 77, 255),
            (153, 69, 1),
            (120, 166, 157)
        ]
    }