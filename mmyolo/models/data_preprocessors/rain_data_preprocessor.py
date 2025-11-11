# Modified from mmdet/models/data_preprocessors/data_preprocessor.py
import os
import math
import cv2
import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from torch import Tensor

from mmdet.models.utils import unfold_wo_center
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmyolo.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import ConfigType


@MODELS.register_module()
class RainDetDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It supports batch augmentations.
    2. It will additionally append batch_input_shape and pad_shape
    to data_samples considering the object detection task.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert the ``BaseBoxes`` type of
            bboxes data to ``Tensor`` type. Defaults to True.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor

        snow_path = '/mnt/dataset/zq/Snow100K/test/Snow100K-M/mask'
        snow_list = sorted(os.listdir(snow_path))
        self.snow_list = [os.path.join(snow_path, img_p) for img_p in snow_list]

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': self.apply_random_effect(inputs), 'gts': inputs, 'data_samples': data_samples}

    def get_noise(img, value=10):
        '''
        #生成噪声图像
        >>> 输入： img图像
            value= 大小控制雨滴的多少
        >>> 返回图像大小的模糊噪声图像
        '''

        noise = np.random.uniform(0, 256, img.shape[0:2])
        # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
        v = value * 0.01
        noise[np.where(noise < (256 - v))] = 0

        # 噪声做初次模糊
        k = np.array([[0, 0.1, 0],
                    [0.1, 8, 0.1],
                    [0, 0.1, 0]])

        noise = cv2.filter2D(noise, -1, k)

        # 可以输出噪声看看
        '''cv2.imshow('img',noise)
        cv2.waitKey()
        cv2.destroyWindow('img')'''
        return noise
    
    def rain_blur(noise, length=10, angle=0, w=1):
        '''
        将噪声加上运动模糊,模仿雨滴

        # >>>输入
        noise：输入噪声图，shape = img.shape[0:2]
        length: 对角矩阵大小，表示雨滴的长度
        angle： 倾斜的角度，逆时针为正
        w:      雨滴大小

        # >>>输出带模糊的噪声

        '''

        # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正

        trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
        dig = np.diag(np.ones(length))  # 生成对焦矩阵

        k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
        k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

        # k = k / length                         #是否归一化

        blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

        # 转换到0-255区间
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        rain = np.expand_dims(blurred, 2)
        blurred = np.repeat(rain, 3, 2)
        # cv2.imwrite('./rain_mask.png',blurred)
        '''
        cv2.imshow('img',blurred)
        cv2.waitKey()
        cv2.destroyWindow('img')'''

        return blurred
    
    def add_rain_vectorized(self, img_batch):
        """
        Apply rain effect to batch of images in vectorized form
        Args:
            img_batch: torch.Tensor, shape [B, C, H, W], range [0, 1]
        Returns:
            torch.Tensor: Images with rain effect
        """
        B, C, H, W = img_batch.shape
        result = img_batch.clone()
        
        # Process each image in the batch
        for i in range(B):
            # Convert image to numpy for OpenCV operations
            img_np = (result[i].permute(1, 2, 0).cpu().numpy()) * 255.

            w = np.random.choice([3, 5, 7, 9, 11], p=[0.2, 0.2, 0.2, 0.2, 0.2])  # 粗细
            length = np.random.randint(20, 41)  # 长度
            angle = np.random.randint(-45, 45)  # 角度

            noise = self.get_noise(img_np, value=np.random.randint(50, 100))
            rain = self.rain_blur(noise, length=length, angle=angle, w=w)

            img_np = img_np + rain
            
            # Convert back to tensor
            result[i] = torch.from_numpy(np.clip(img_np, 0, 255) / 255.).permute(2, 0, 1).to(img_batch.device)
            
        return result
    
    def apply_random_effect(self, img_batch):
        """
        Randomly apply one of the weather effects with probability effect_prob
        Args:
            img_batch: torch.Tensor, shape [B, C, H, W], range [0, 1]
        Returns:
            torch.Tensor: Images with randomly applied effects
        """        
        return self.add_rain_vectorized(img_batch)
    
    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[3] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape

    def pad_gt_masks(self,
                     batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape,
                    pad_val=self.mask_pad_value)

    def pad_gt_sem_seg(self,
                       batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_sem_seg to shape of batch_input_shape."""
        if 'gt_sem_seg' in batch_data_samples[0]:
            for data_samples in batch_data_samples:
                gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                h, w = gt_sem_seg.shape[-2:]
                pad_h, pad_w = data_samples.batch_input_shape
                gt_sem_seg = F.pad(
                    gt_sem_seg,
                    pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                    mode='constant',
                    value=self.seg_pad_value)
                data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)


@MODELS.register_module()
class YOLOv5RainDetDataPreprocessor(RainDetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolov5_collate`
    """

    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        super().__init__(*args, non_blocking=non_blocking, **kwargs)

    def get_noise(self, img, value=10):
        '''
        #生成噪声图像
        >>> 输入： img图像
            value= 大小控制雨滴的多少
        >>> 返回图像大小的模糊噪声图像
        '''

        noise = np.random.uniform(0, 256, img.shape[0:2])
        # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
        v = value * 0.01
        noise[np.where(noise < (256 - v))] = 0

        # 噪声做初次模糊
        k = np.array([[0, 0.1, 0],
                    [0.1, 8, 0.1],
                    [0, 0.1, 0]])

        noise = cv2.filter2D(noise, -1, k)

        # 可以输出噪声看看
        '''cv2.imshow('img',noise)
        cv2.waitKey()
        cv2.destroyWindow('img')'''
        return noise
    
    def rain_blur(self, noise, length=10, angle=0, w=1):
        '''
        将噪声加上运动模糊,模仿雨滴

        # >>>输入
        noise：输入噪声图，shape = img.shape[0:2]
        length: 对角矩阵大小，表示雨滴的长度
        angle： 倾斜的角度，逆时针为正
        w:      雨滴大小

        # >>>输出带模糊的噪声

        '''

        # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正

        trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
        dig = np.diag(np.ones(length))  # 生成对焦矩阵

        k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
        k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

        # k = k / length                         #是否归一化

        blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

        # 转换到0-255区间
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        rain = np.expand_dims(blurred, 2)
        blurred = np.repeat(rain, 3, 2)
        # cv2.imwrite('./rain_mask.png',blurred)
        '''
        cv2.imshow('img',blurred)
        cv2.waitKey()
        cv2.destroyWindow('img')'''

        return blurred
    
    def add_rain_vectorized(self, img_batch):
        """
        Apply rain effect to batch of images in vectorized form
        Args:
            img_batch: torch.Tensor, shape [B, C, H, W], range [0, 1]
        Returns:
            torch.Tensor: Images with rain effect
        """
        B, C, H, W = img_batch.shape
        result = img_batch.clone()
        
        # Process each image in the batch
        for i in range(B):
            # Convert image to numpy for OpenCV operations
            img_np = (result[i].permute(1, 2, 0).cpu().numpy()) * 255.

            w = np.random.choice([3, 5, 7, 9, 11], p=[0.2, 0.2, 0.2, 0.2, 0.2])  # 粗细
            length = np.random.randint(20, 41)  # 长度
            angle = np.random.randint(-45, 45)  # 角度

            noise = self.get_noise(img_np, value=np.random.randint(50, 100))
            rain = self.rain_blur(noise, length=length, angle=angle, w=w)

            img_np = img_np + rain
            
            # Convert back to tensor
            result[i] = torch.from_numpy(np.clip(img_np, 0, 255) / 255.).permute(2, 0, 1).to(img_batch.device)
            
        return result
    
    def apply_random_effect(self, img_batch):
        """
        Randomly apply one of the weather effects with probability effect_prob
        Args:
            img_batch: torch.Tensor, shape [B, C, H, W], range [0, 1]
        Returns:
            torch.Tensor: Images with randomly applied effects
        """        
        return self.add_rain_vectorized(img_batch)
    
    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return super().forward(data, training)

        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        assert isinstance(data['data_samples'], dict)

        # TODO: Supports multi-scale training
        if self._channel_conversion and inputs.shape[1] == 3:
            inputs = inputs[:, [2, 1, 0], ...]
        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)
        
        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples_output = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'img_metas': img_metas
        }
        if 'masks' in data_samples:
            data_samples_output['masks'] = data_samples['masks']
        if 'keypoints' in data_samples:
            data_samples_output['keypoints'] = data_samples['keypoints']
            data_samples_output['keypoints_visible'] = data_samples[
                'keypoints_visible']

        return {'inputs': self.apply_random_effect(inputs), 'gts': inputs, 'data_samples': data_samples_output}