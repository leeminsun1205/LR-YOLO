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
from .yolo_haze_dehaze_detector import YOLOHazeDehazeDetector

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOHazeSFNetDetector(YOLOHazeDehazeDetector):
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
                 use_syncbn: bool = True):
        super().__init__(
            dehaze_model=dehaze_model,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
            use_syncbn=use_syncbn)
    
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
        # x = self.extract_feat(dehaze_outputs)
        # losses = self.bbox_head.loss(x, batch_data_samples)
        batch_gts2 = F.interpolate(batch_gts, scale_factor=0.5, mode='bilinear')
        batch_gts4 = F.interpolate(batch_gts, scale_factor=0.25, mode='bilinear')
        losses = {}
        losses['loss_dehaze_l1'] = self.criterion_l1(dehaze_outputs[2], batch_gts)
        losses['loss_dehaze_l1_x2'] = self.criterion_l1(dehaze_outputs[1], batch_gts2)
        losses['loss_dehaze_l1_x4'] = self.criterion_l1(dehaze_outputs[0], batch_gts4)
        return losses
    
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        dehaze_outputs = self.dehaze_model(batch_inputs)
        x = self.extract_feat(dehaze_outputs[-1])
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        # add input and output image to data_samples for better visualization
        for batch_data_sample, input, output in zip(batch_data_samples, batch_inputs, dehaze_outputs[-1]):
            # rescale input and output in original image space
            ori_h, ori_w = batch_data_sample.ori_shape[:2]
            pad_param = batch_data_sample.pad_param.astype(np.int32)
            crop_y1, crop_y2 = pad_param[0], input.shape[-2] - pad_param[1]
            crop_x1, crop_x2 = pad_param[2], input.shape[-1] - pad_param[3]
            input = input[..., crop_y1:crop_y2, crop_x1:crop_x2]
            output = output[..., crop_y1:crop_y2, crop_x1:crop_x2]
            input = F.interpolate(input.unsqueeze(0), size=[ori_h, ori_w], mode='bilinear', align_corners=False)[0]
            output = F.interpolate(output.unsqueeze(0), size=[ori_h, ori_w], mode='bilinear', align_corners=False)[0]
            batch_data_sample.image = PixelData(input=input, output=output)
        
        return batch_data_samples
    
    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        dehaze_outputs = self.dehaze_model(batch_inputs)
        x = self.extract_feat(dehaze_outputs[-1])
        results = self.bbox_head.forward(x)
        return results