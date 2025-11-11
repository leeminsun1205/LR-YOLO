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


class GatedDIP(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self,
                encoder_output_dim : int = 256,
                num_of_gates : int = 7):

        super(GatedDIP,self).__init__()
        # Encoder Model
        self.encoder = torchvision.models.vgg16(pretrained=False)
        # Changed 4096 --> 256 dimension
        self.encoder.classifier[6] = torch.nn.Linear(4096,encoder_output_dim,bias=True)


        # DIP Controller Module
        # self.dip_controller = torch.nn.LSTMCell(lstm_input_dim,lstm_hidden_dim)

        # # Dropout
        # self.dropout = torch.nn.Dropout(p=0.1)

        # Gating Module
        self.gate_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,num_of_gates,bias=True))
        
        # White-Balance Module
        self.wb_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,3,bias=True))
        
        # Gamma Module
        self.gamma_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        
        # Sharpning Module
        self.gaussian_blur = torchvision.transforms.GaussianBlur(13, sigma=(0.1, 5.0))
        self.sharpning_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))

        # De-Fogging Module
        self.defogging_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))

        # Contrast Module
        self.contrast_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))

        # Contrast Module
        self.tone_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,8,bias=True))
        
        # Initializing layer with xavier uniform distribution.
        self.gamma_module.apply(self._init_weights)
        self.defogging_module.apply(self._init_weights)
        self.sharpning_module.apply(self._init_weights)
        self.wb_module.apply(self._init_weights)
        self.contrast_module.apply(self._init_weights)
        self.tone_module.apply(self._init_weights)
    
    def _init_weights(self,m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    def rgb2lum(self,img):
        img = 0.27 * img[:, 0, :, :] + 0.67 * img[:, 1, :,:] + 0.06 * img[:, 2, :, :]
        return img 
    
    def lerp(self ,a, b, l):
        #print('a',a.shape,'b',b.shape,'l',l.shape)
        return (1 - l.unsqueeze(2).unsqueeze(3)) * a + l.unsqueeze(2).unsqueeze(3) * b


    def dark_channel(self,x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        z = x.min(dim=1)[0].unsqueeze(1)
        return torch.cat((z,z,z),dim=1)
    
    def atmospheric_light(self,x : torch.tensor ,top_k : int=1000):
        """_summary_

        Args:
            x (_type_): _description_
            top_k (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        r,_ = torch.sort(x[:,0,:,:].view(x.shape[0],-1),descending=True)
        g,_ = torch.sort(x[:,1,:,:].view(x.shape[0],-1),descending=True)
        b,_ = torch.sort(x[:,2,:,:].view(x.shape[0],-1),descending=True)
        r_value = r[:,:top_k].mean(dim=1).unsqueeze(1)
        g_value = g[:,:top_k].mean(dim=1).unsqueeze(1)
        b_value = b[:,:top_k].mean(dim=1).unsqueeze(1)
        a = torch.cat((r_value,g_value,b_value),dim=1).unsqueeze(2).unsqueeze(3)
        return a
    
    def blur(self,x : torch.tensor):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.gaussian_blur(x)


    def defog(self,x:torch.tensor ,latent_out : torch.tensor ,fog_gate : torch.tensor):
        """Defogging module is used for removing the fog from the image using ASM 
        (Atmospheric Scattering Model).
        I(X) = (1-T(X)) * J(X) + T(X) * A(X)
        I(X) => image containing the fog.
        T(X) => Transmission map of the image.
        J(X) => True image Radiance.
        A(X) => Atmospheric scattering factor.

        Args:
            x (torch.tensor): Input image I(X)
            latent_out (torch.tensor): Feature representation from DIP Module.
            fog_gate (torch.tensor): Gate value raning from (0. - 1.) which enables defog module.

        Returns:
            torch.tensor : Returns defogged image with true image radiance.
        """
        omega = self.defogging_module(latent_out).unsqueeze(2).unsqueeze(3)
        omega = self.tanh_range(omega,torch.tensor(0.1),torch.tensor(1.))
        
        # print('i',i.min(),i.max())
        a = torch.maximum(self.atmospheric_light(x),torch.tensor(0.01))
        i = x/a 
        i = self.dark_channel(i)
        t = 1. - (omega*i)
        j = ((x-a)/(torch.maximum(t,torch.tensor(0.01))))+a
        j = (j - j.min())/(j.max()-j.min())
        j = j* fog_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return j
        
    def white_balance(self,x : torch.tensor,latent_out : torch.tensor ,wb_gate: torch.tensor):
        """_summary_

        Args:
            x (_type_): _description_
            latent_out (_type_): _description_
            wb_gate (_type_): _description_

        Returns:
            _type_: _description_
        """
        log_wb_range = 0.5
        wb = self.wb_module(latent_out)
        wb = torch.exp(self.tanh_range(wb,-log_wb_range,log_wb_range))
        
        color_scaling = 1./(1e-5 + 0.27 * wb[:, 0] + 0.67 * wb[:, 1] +
        0.06 * wb[:, 2])
        wb = color_scaling.unsqueeze(1)*wb
        wb_out = wb.unsqueeze(2).unsqueeze(3)*x
        wb_out = (wb_out-wb_out.min())/(wb_out.max()-wb_out.min())
        wb_out = wb_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)*wb_out
        return wb_out

    def tanh01(self,x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return torch.tanh(x)*0.5+0.5

    def tanh_range(self,x,left,right):
        """_summary_

        Args:
            x (_type_): _description_
            left (_type_): _description_
            right (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.tanh01(x)*(right-left)+ left

    def gamma_balance(self,x : torch.tensor,latent_out : torch.tensor,gamma_gate : torch.tensor):
        """_summary_

        Args:
            x (_type_): _description_
            latent_out (_type_): _description_
            gamma_gate (_type_): _description_

        Returns:
            _type_: _description_
        """
        log_gamma = torch.log(torch.tensor(2.5))
        gamma = self.gamma_module(latent_out).unsqueeze(2).unsqueeze(3)
        gamma = torch.exp(self.tanh_range(gamma,-log_gamma,log_gamma))
        g = torch.pow(torch.maximum(x,torch.tensor(1e-4)),gamma)
        g = (g-g.min())/(g.max()-g.min())
        g = g*gamma_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return g
    
    def sharpning(self,x,latent_out,sharpning_gate):
        """_summary_

        Args:
            x (_type_): _description_
            latent_out (_type_): _description_
            sharpning_gate (_type_): _description_

        Returns:
            _type_: _description_
        """
        out_x = self.blur(x)
        y = self.sharpning_module(latent_out).unsqueeze(2).unsqueeze(3)
        y = self.tanh_range(y,torch.tensor(0.1),torch.tensor(1.))
        s = x + (y*(x-out_x))
        s = (s-s.min())/(s.max()-s.min())
        s = s * (sharpning_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        return s
    
    def identity(self,x : torch.tensor,identity_gate : torch.tensor):
        """_summary_

        Args:
            x (_type_): _description_
            identity_gate (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = x*identity_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return x
    
    def contrast(self,x,latent_out,contrast_gate):
        alpha = torch.tanh(self.contrast_module(latent_out))
        luminance = torch.minimum(torch.maximum(self.rgb2lum(x), torch.tensor(0.0)), torch.tensor(1.0)).unsqueeze(1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = x / (luminance + 1e-6) * contrast_lum
        contrast_image = self.lerp(x, contrast_image, alpha) 
        contrast_image = (contrast_image-contrast_image.min())/(contrast_image.max()-contrast_image.min())
        contrast_image = contrast_image * contrast_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return contrast_image
    
    def tone(self,x,latent_out,tone_gate):
        curve_steps = 8
        tone_curve = self.tone_module(latent_out).reshape(-1,1,curve_steps)
        tone_curve = self.tanh_range(tone_curve,0.5, 2)
        tone_curve_sum = torch.sum(tone_curve, dim=2) + 1e-30
        total_image = x * 0
        for i in range(curve_steps):
            total_image += torch.clamp(x - 1.0 * i /curve_steps, 0, 1.0 /curve_steps) \
                            * tone_curve[:,:,i].unsqueeze(2).unsqueeze(3)
        total_image *= curve_steps / tone_curve_sum.unsqueeze(2).unsqueeze(3)
        total_image = (total_image-total_image.min())/(total_image.max()-total_image.min())
        total_image = total_image * tone_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return total_image


    
    def forward(self, x : torch.Tensor):
        latent_out = torch.nn.functional.relu_(self.encoder(F.interpolate(x, size=[256, 256], mode='bilinear')))
        gate = self.tanh_range(self.gate_module(latent_out),0.01,1.0)
        wb_out = self.white_balance(x,latent_out,gate[:,0])
        gamma_out = self.gamma_balance(x,latent_out,gate[:,1])
        identity_out = self.identity(x,gate[:,2])
        sharpning_out = self.sharpning(x,latent_out,gate[:,3])
        fog_out = self.defog(x,latent_out,gate[:,4])
        contrast_out = self.contrast(x,latent_out,gate[:,5])
        tone_out = self.tone(x,latent_out,gate[:,6])
        x = wb_out + gamma_out   + fog_out + sharpning_out + contrast_out + tone_out + identity_out
        x = (x-x.min())/(x.max()-x.min())
        return x,gate


@MODELS.register_module()
class YOLOHazeGDIPDetector(SingleStageDetector):
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
        
        self.gated_dip = GatedDIP(256)

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
        ia_outputs, _ = self.gated_dip(batch_inputs)
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