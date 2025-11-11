import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Dict, List, Tuple, Union
from torch import Tensor
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmengine.structures import PixelData
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmyolo.models.losses import ContrastLoss_res

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

from mmyolo.registry import MODELS


class ScaleAwareFeatureAggregation(nn.Module):

    """
      ScaleAwareFeatureAggregation happens in five steps

      1 : Downgrade the size of two feature maps.
      2 : Converts the feature maps into query and key representaions.
      3 : Concats the downgraded features and divide it into blocks.
      4 : Performs Multi-Head Attention.
      5 : Aggregates and outputs the enhanced representation

    """


    def __init__(self, channels):
        super().__init__()

        self.mult_scale_heads = 8

        self.query_conv1 = nn.Conv2d(in_channels = channels , out_channels = channels , kernel_size= 7, stride=4)
        self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)
        self.key_conv = nn.Conv2d(in_channels = channels , out_channels = channels , kernel_size=3, stride=2)


    def forward(self, x, quarter_scale_x):
        """ 
        Args : 
            x:  a list of, feature map in (C, H, W) format.
            quarter_scale_x: a list of, quarter scale feature map in (C, H/4, W/4) format.

        Returns :
            aggregated_enhanced_representation : enhanced aggregated feature representations of
                                                 two feature maps from different scales.
                                                 a list of, enhanced features in (C, H, W)

        """

        orig_x = x

        # Key Query Generation
        x = self.query_conv1(x)
        x = self.query_conv2(x)
        quarter_scale_x = self.key_conv(quarter_scale_x)
        batch_size, C, roi_h, roi_w = x.size()
        
        x = x.view(batch_size, 1, C, roi_h, roi_w)
        quarter_scale_x = quarter_scale_x.view(batch_size, 1, C, roi_h, roi_w)

        x = torch.cat((x, quarter_scale_x), dim=1)
        batch_size, img_n, _, roi_h, roi_w = x.size()

        # Calculating the number of attention blocks
        x_embed = x
        c_embed = x_embed.size(2)

        # Performing multi-head attention
        # (img_n, num_attention_blocks, C / num_attention_blocks, H, W)
        x_embed = x_embed.view(batch_size, img_n, self.mult_scale_heads, -1, roi_h,
                               roi_w)
        # (1, roi_n, num_attention_blocks, C / num_attention_blocks, H, W)
        target_x_embed = x_embed[:, [1]]
        # (batch_size, img_n, num_attention_blocks, 1, H, W)
        ada_weights = torch.sum(
            x_embed * target_x_embed, dim=3, keepdim=True) / (
                float(c_embed / self.mult_scale_heads)**0.5)
        # (batch_size, img_n, num_attention_blocks, C / num_attention_blocks, H, W)
        ada_weights = ada_weights.expand(-1, -1, -1,
                                         int(c_embed / self.mult_scale_heads),
                                         -1, -1).contiguous()
        ada_weights = ada_weights.view(batch_size, img_n, c_embed, roi_h, roi_w)
        ada_weights = ada_weights.softmax(dim=1)

        # Aggregation and generation of enhanced representation
        x = (x * ada_weights).sum(dim=1)
        upsample = nn.UpsamplingBilinear2d((orig_x.size()[-2], orig_x.size()[-1]))
        aggregated_feature = upsample(x)
        aggregated_enhanced_representation = orig_x + aggregated_feature
        return aggregated_enhanced_representation

class FeatEnHancer(nn.Module):

    """
      Feature extraction for dark objects occurs in three stages
      1 : Extract low light features in three scales 
      2 : Scale aware aggregation of low light features 
      3 : Construction of enhanced representation of the input image


      Args : 
           in_channels : represent the number of channels in the input image (default is 3, considering RGB image)

    """

    def __init__(self, in_channels=3):
        super(FeatEnHancer, self).__init__()

        int_out_channels = 32
        out_channels = 24

        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(in_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(int_out_channels, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(int_out_channels * 2, int_out_channels, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(int_out_channels * 2, out_channels, 3, 1, 1, bias=True)

        # our convolution layers to transform the concatenated feature maps into the required feature shapes
        self.ue_conv8 = nn.Conv2d(out_channels*2, int_out_channels, 3, 1, 1, bias=True)
        self.ue_conv9 = nn.Conv2d(int_out_channels, out_channels, 3, 1, 1, bias=True)

        # Convolutions for downsampling orignal image into multiple scales
        self.quarter_conv = nn.Conv2d(in_channels, in_channels, 7, 4)
        self.hexa_conv = nn.Conv2d(in_channels, in_channels, 3, 2)

        self.scale_aware_aggregation = ScaleAwareFeatureAggregation(channels=24).to('cuda')

    def forward(self, x):

        """
        Args:
            x : a list, batched inputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                * image: Tensor, image in (C, H, W) format.
        
        Returns:
            enhanced_representation : a list, of enhanced representations
                                      Each representation is a Tensor, in (N, C, H, W) format.  
        """

        quarter_scale_x = self.quarter_conv(x)
        hexa_scale_x = self.hexa_conv(quarter_scale_x)
       
        # Extracting low light featurs at three different scales
        x1 = self.relu(self.e_conv1(x))
        quarter_scale_x1 = self.relu(self.e_conv1(quarter_scale_x))
        hexa_scale_x1 = self.relu(self.e_conv1(hexa_scale_x))

        x2 = self.relu(self.e_conv2(x1))
        quarter_scale_x2 = self.relu(self.e_conv2(quarter_scale_x1))
        hexa_scale_x2 = self.relu(self.e_conv2(hexa_scale_x1))

        x3 = self.relu(self.e_conv3(x2))
        quarter_scale_x3 = self.relu(self.e_conv3(quarter_scale_x2))
        hexa_scale_x3 = self.relu(self.e_conv3(hexa_scale_x2))

        x4 = self.relu(self.e_conv4(x3))
        quarter_scale_x4 = self.relu(self.e_conv4(quarter_scale_x3))
        hexa_scale_x4 = self.relu(self.e_conv4(hexa_scale_x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        quarter_scale_x5 = self.relu(self.e_conv5(torch.cat([quarter_scale_x3, quarter_scale_x4], 1)))
        hexa_scale_x5 = self.relu(self.e_conv5(torch.cat([hexa_scale_x3, hexa_scale_x4], 1)))

        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        quarter_scale_x6 = self.relu(self.e_conv6(torch.cat([quarter_scale_x2, quarter_scale_x5], 1)))
        hexa_scale_x6 = self.relu(self.e_conv6(torch.cat([hexa_scale_x2, hexa_scale_x5], 1)))

        x7 = self.relu(self.e_conv7(torch.cat([x1, x6], 1)))
        quarter_scale_x7 = self.relu(self.e_conv7(torch.cat([quarter_scale_x1, quarter_scale_x6], 1)))
        hexa_scale_x7 = self.e_conv7(torch.cat([hexa_scale_x1, hexa_scale_x6], 1))

        # Applying ScaleAwareFeatureAggregation between X7 and quarter_scale_x7
        x7 = self.feature_aggregation_block(x7, quarter_scale_x7)

        # Upsampling hexa scale to H x W for Skip connection
        x_upsample = nn.UpsamplingBilinear2d((x7.size()[-2], x7.size()[-1]))
        hexa_scale_x7 = x_upsample(hexa_scale_x7)
        x8 = self.ue_conv8(torch.cat([x7, hexa_scale_x7], 1))

        # Construction of Enhanced representation
        activated_enhanced_feature = torch.tanh(self.ue_conv9(x8))
        feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8 = torch.split(activated_enhanced_feature, 3, dim=1)  
        x = x + feature1
        x = x + feature2
        x = x + feature3
        x = x + feature4
        x = x + feature5
        x = x + feature6
        x = x + feature7
        enhanced_representation = x + feature8
        return enhanced_representation
        
    def feature_aggregation_block(self, x, quarter_scale_x):

        # Does the forward pass and returns the enhanced feature representation by aggregating both input feature maps
        """
        Args:
            x:  a feature map in (C, H, W) format.
            quarter_scale_x:  quarter scale feature map in (C, H/4, W/4) format.
        
        Returns:
            aggregated_enhanced_representation : Tensor, in (C, H, W) format.
        """
        aggregated_enhanced_representation = self.scale_aware_aggregation(x, quarter_scale_x)
        return aggregated_enhanced_representation


@MODELS.register_module()
class YOLODarkFeatEnHancerDetector(SingleStageDetector):
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
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.feat_enhancer = FeatEnHancer(3)
        
        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')
    
    def loss(self, batch_inputs: Tensor,
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
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
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
        x, dehaze_outputs = self.extract_feat(batch_inputs, return_outputs=True)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        # add input and output image to data_samples for better visualization
        for batch_data_sample, input, output in zip(batch_data_samples, batch_inputs, dehaze_outputs):
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
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results
    
    def extract_feat(self, batch_inputs: Tensor, return_outputs: bool = False) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        # Image-Adaptive
        ia_outputs = self.feat_enhancer(batch_inputs)
        batch_inputs = ia_outputs

        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        
        if return_outputs:
            return x, ia_outputs
        return x
    
    def forward(self,
                inputs: torch.Tensor,
                gts: torch.Tensor = None,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')