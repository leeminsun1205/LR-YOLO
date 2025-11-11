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
class YOLOHazeSFNetDetectorAD(YOLOHazeSFNetDetector):
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

        # self.backbone.train()
        # self.neck.train()
        # self.bbox_head.train()
    
    def get_attack(self, batch_gts: Tensor, batch_data_samples: SampleList, eps=4/255., eps_iter=0.5/255., n_iter=10) -> Tensor:
        eta = torch.from_numpy(np.random.uniform(-eps, eps, size=batch_gts.shape)).type(torch.FloatTensor).to(batch_gts.device)
        eta = eta.detach()
        x_adv = torch.clip(batch_gts + eta, 0.0, 1.0).detach()

        beta1 = 0.9
        beta2 = 0.999
        momentum = 0
        vol = 0
        # prev_loss = 1.

        for t in range(1, 1+n_iter):
            x_adv_t = x_adv.clone().detach().requires_grad_(True)

            # 前向计算
            x = self.extract_feat(x_adv_t)
            losses = self.bbox_head.loss(x, batch_data_samples)
            loss = sum(losses.values())
            # 反向传播获取梯度
            loss.backward()
            grad = x_adv_t.grad.detach()

            with torch.no_grad():
                # 更新动量与二阶动量
                momentum = beta1 * momentum + (1 - beta1) * grad
                vol = beta2 * vol + (1 - beta2) * grad**2
                m_hat = momentum / (1 - beta1**t)
                v_hat = vol / (1 - beta2**t)
                
                # 计算自适应学习率梯度
                grad = m_hat / (torch.sqrt(v_hat) + 1e-8)
                signed_grad = torch.sign(grad)
                
                # 更新对抗样本
                x_adv -= eps_iter * signed_grad
                eta = torch.clamp(x_adv - batch_gts, min=-eps, max=eps)
                x_adv = torch.clamp(batch_gts + eta, 0.0, 1.0)
                
                # 提前终止检查
                # if abs(loss.item() - prev_loss) / prev_loss < 1e-3:
                #     break
                # prev_loss = loss.item()
        
        return x_adv.detach()
    
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

        batch_gts = self.get_attack(batch_gts, batch_data_samples)
        batch_gts2 = F.interpolate(batch_gts, scale_factor=0.5, mode='bilinear')
        batch_gts4 = F.interpolate(batch_gts, scale_factor=0.25, mode='bilinear')
        losses['loss_dehaze_l1_ad'] = self.criterion_l1(dehaze_outputs[2], batch_gts)
        losses['loss_dehaze_l1_ad_x2'] = self.criterion_l1(dehaze_outputs[1], batch_gts2)
        losses['loss_dehaze_l1_ad_x4'] = self.criterion_l1(dehaze_outputs[0], batch_gts4)

        return losses