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
from ..utils.filters import DefogFilter, ImprovedWhiteBalanceFilter, GammaFilter, ToneFilter, ContrastFilter, UsmFilter
from easydict import EasyDict as edict

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

from mmyolo.registry import MODELS


def DarkChannel(im):
    R = im[0, :, :]
    G = im[1, :, :]
    B = im[2, :, :]
    dc = torch.min(torch.min(R, G), B)
    return dc


def AtmLight(im, dark):
    c, h, w = im.shape
    imsz = h * w
    numpx = int(max(torch.floor(torch.tensor(imsz) / 1000), torch.tensor(1)))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(3, imsz)

    indices = torch.argsort(darkvec)
    indices = indices[(imsz - numpx):imsz]

    atmsum = torch.zeros([3, 1]).to(imvec.device)
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[:, indices[ind]]

    A = atmsum / numpx
    return A.reshape(1, 3)


def DarkIcA(im, A):
    c, h, w = im.shape
    im3 = torch.zeros([c,h,w]).to(im.device)
    for ind in range(0, 3):
        im3[ind, :, :] = im[ind, :, :] / A[0, ind]
    return DarkChannel(im3)


class Dip_Cnn(nn.Module):
    def __init__(self):
        super(Dip_Cnn, self).__init__()
        channels = 16
        self.cnnnet = nn.Sequential(
                        nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
                        
                        nn.Conv2d(channels, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
                        
                        nn.Conv2d(channels*2, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
                        
                        nn.Conv2d(channels*2, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
                        
                        nn.Conv2d(channels*2, channels*2, kernel_size=3, stride=2, padding=1, bias=True),
                        nn.LeakyReLU(negative_slope=0.1),
        )
        self.full_layers1 = nn.Sequential(
                        nn.Linear(2048, 64),
                        nn.Linear(64, 15),
        )
    def forward(self, x):
        out = self.cnnnet(x)
        out = out.view(out.size(0), -1)
        out = self.full_layers1(out)
        return out


@MODELS.register_module()
class YOLOHazeIADetector(SingleStageDetector):
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
        
        self.dip_filters = Dip_Cnn()
        self.cfg = edict() 
        self.cfg.filters = [
            DefogFilter, ImprovedWhiteBalanceFilter,  GammaFilter,
            ToneFilter, ContrastFilter, UsmFilter
        ]
        self.cfg.num_filter_parameters = 15
        self.cfg.defog_begin_param = 0
        self.cfg.wb_begin_param = 1
        self.cfg.gamma_begin_param = 4
        self.cfg.tone_begin_param = 5
        self.cfg.contrast_begin_param = 13
        self.cfg.usm_begin_param = 14
        self.cfg.curve_steps = 8
        self.cfg.gamma_range = 3
        self.cfg.exposure_range = 3.5
        self.cfg.wb_range = 1.1
        self.cfg.color_curve_range = (0.90, 1.10)
        self.cfg.lab_curve_range = (0.90, 1.10)
        self.cfg.tone_curve_range = (0.5, 2)
        self.cfg.defog_range = (0.1, 1.0)
        self.cfg.usm_range = (0.0, 5)

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
        resized_batch_inputs = F.interpolate(batch_inputs, size=[256, 256], mode='bilinear')
        features = self.dip_filters(resized_batch_inputs)

        filtered_image_batch = batch_inputs
        B, C, W, H = batch_inputs.shape
        #----------------
        #构建defog的过滤器参数
        #----------------
        dark = torch.zeros([B, W, H],dtype=torch.float32).to(batch_inputs.device)   
        defog_A = torch.zeros([B, C],dtype=torch.float32).to(batch_inputs.device)   
        IcA = torch.zeros([B, W, H],dtype=torch.float32).to(batch_inputs.device)
        for i in range(B):
            dark_i = DarkChannel(batch_inputs[i])
            defog_A_i = AtmLight(batch_inputs[i], dark_i)
            IcA_i = DarkIcA(batch_inputs[i], defog_A_i)
            dark[i, ...] = dark_i
            defog_A[i, ...] = defog_A_i
            IcA[i, ...] = IcA_i
        IcA = IcA.unsqueeze(-1)
        #需要经过的6个过滤器
        filters = self.cfg.filters
        filters = [x(filtered_image_batch, self.cfg) for x in filters]
        filter_features = features
        for j, filter in enumerate(filters):

            filtered_image_batch, filter_parameter = filter.apply(
                filtered_image_batch, filter_features, defog_A, IcA)

        # yolo
        ia_outputs = torch.clamp(filtered_image_batch, 0, 1)
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