import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union
from torch import Tensor
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmengine.structures import PixelData
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmyolo.models.losses import ContrastLoss_res
from .yolo_haze_sfnet_detector import YOLOHazeSFNetDetector

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOHazeSFNetDetectorSR4IR(YOLOHazeSFNetDetector):
    r"""Implementation of YOLO Series

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 dehaze_model: ConfigType,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 ):
        super().__init__(
            dehaze_model=dehaze_model,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.cnt = 0
        self.backbone.train()
        self.neck.train()
        self.bbox_head.train()
    
    def loss(self, batch_inputs: Tensor, batch_gts: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        dehaze_outputs = self.dehaze_model(batch_inputs)

        # phase 1 : update network dehaze, freeze network detection
        if self.cnt % 2 == 0:
            # frozen detector
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.neck.parameters():
                param.requires_grad = False
            for param in self.bbox_head.parameters():
                param.requires_grad = False
            batch_gts2 = F.interpolate(batch_gts, scale_factor=0.5, mode='bilinear')
            batch_gts4 = F.interpolate(batch_gts, scale_factor=0.25, mode='bilinear')
            losses = {}
            losses['loss_dehaze_l1'] = self.criterion_l1(dehaze_outputs[2], batch_gts)
            losses['loss_dehaze_l1_x2'] = self.criterion_l1(dehaze_outputs[1], batch_gts2)
            losses['loss_dehaze_l1_x4'] = self.criterion_l1(dehaze_outputs[0], batch_gts4)
            x1 = self.backbone(dehaze_outputs[-1])
            x2 = self.backbone(batch_gts)
            losses['loss_dehaze_tdp'] = self.criterion_l1(x1[-1], x2[-1])
        # phase 2 : update network detection, freeze network dehaze
        else:
            # train detector
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.neck.parameters():
                param.requires_grad = True
            for param in self.bbox_head.parameters():
                param.requires_grad = True
            x = self.extract_feat(dehaze_outputs[-1].detach())
            losses = self.bbox_head.loss(x, batch_data_samples)
        
        self.cnt += 1
        return losses


@MODELS.register_module()
class YOLOHazeSFNetDetectorSR4IR1st(YOLOHazeSFNetDetector):
    r"""Implementation of YOLO Series

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 dehaze_model: ConfigType,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 ):
        super().__init__(
            dehaze_model=dehaze_model,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.backbone.train()
        self.neck.train()
        self.bbox_head.train()
        # frozen detector
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.neck.parameters():
            param.requires_grad = False
        for param in self.bbox_head.parameters():
            param.requires_grad = False
    
    def loss(self, batch_inputs: Tensor, batch_gts: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        dehaze_outputs = self.dehaze_model(batch_inputs)

        # phase 1 : update network dehaze, freeze network detection
        batch_gts2 = F.interpolate(batch_gts, scale_factor=0.5, mode='bilinear')
        batch_gts4 = F.interpolate(batch_gts, scale_factor=0.25, mode='bilinear')
        losses = {}
        losses['loss_dehaze_l1'] = self.criterion_l1(dehaze_outputs[2], batch_gts)
        losses['loss_dehaze_l1_x2'] = self.criterion_l1(dehaze_outputs[1], batch_gts2)
        losses['loss_dehaze_l1_x4'] = self.criterion_l1(dehaze_outputs[0], batch_gts4)
        x1 = self.backbone(dehaze_outputs[-1])
        x2 = self.backbone(batch_gts)
        losses['loss_dehaze_tdp'] = self.criterion_l1(x1[-1], x2[-1])
        
        return losses


@MODELS.register_module()
class YOLOHazeSFNetDetectorSR4IR2nd(YOLOHazeSFNetDetector):
    r"""Implementation of YOLO Series

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 dehaze_model: ConfigType,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 ):
        super().__init__(
            dehaze_model=dehaze_model,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.backbone.train()
        self.neck.train()
        self.bbox_head.train()
        # train detector
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.neck.parameters():
            param.requires_grad = True
        for param in self.bbox_head.parameters():
            param.requires_grad = True
        # frozen dehaze model
        for param in self.dehaze_model.parameters():
            param.requires_grad = False
    
    def loss(self, batch_inputs: Tensor, batch_gts: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        dehaze_outputs = self.dehaze_model(batch_inputs)
        # phase 2 : update network detection, freeze network dehaze
        x = self.extract_feat(dehaze_outputs[-1].detach())
        losses = self.bbox_head.loss(x, batch_data_samples)
        
        return losses