"""Trusted native checkpoint inference; lazy imports, one cached model, one GPU."""
import gc
import copy
import time
from pathlib import Path
import numpy as np
from .core import CLASSES, digest
from ..methods import method_info


def serving_model_config(model):
    """Keep native prediction tensors; exclude only known training constructor options.

    S9 prototype buffers are nonpersistent, so strict state loading still applies.
    No bank file or custom training detector is constructed for serving.
    """
    result=copy.deepcopy(model)
    if result['type']=='PrototypeMemorySelectiveDetector':
        if result.get('sparse_case',8)!=8:
            raise ValueError('Prototype detector must preserve coverage sampling')
        for key in ('prototype_directory','prototype_identity','prototype_bank_sha256',
                    'prototype_weight','prototype_warmup_updates','prototype_cap','prototype_profile'):
            result.pop(key,None)
        result['type']='SelectiveReconstructionDetector'
    if result['type']=='SelectiveReconstructionDetector':
        result['type']='YOLOReconstructDehazeDetector'
        for key in ('sparse_case','crop_count','crop_core','crop_halo','sampling_seed','coverage_candidates'):
            result.pop(key,None)
    if result['type'] not in ('YOLOReconstructDetector','YOLOReconstructDehazeDetector'):
        raise ValueError('Unsupported detector type; use native/selective reconstruction checkpoints')
    result['data_preprocessor']['type']='YOLOv5DetDataPreprocessor'
    result.pop('init_cfg',None)
    result['backbone'].pop('init_cfg',None)
    return result


def serving_pipeline(pipeline):
    """No XML is required for a single uploaded image; preserve eval geometry."""
    result=[copy.deepcopy(t) for t in pipeline
            if t['type'].split('.')[-1] not in ('LoadAnnotations','LoadPanopticAnnotations')]
    if not result or result[0]['type'].split('.')[-1] not in ('LoadImageFromFile','LoadImageFromNDArray'):
        raise ValueError('Unsupported inference image loader')
    result[0]=dict(type='mmdet.LoadImageFromNDArray')
    if result[-1]['type'].split('.')[-1]!='PackDetInputs':raise ValueError('Missing PackDetInputs')
    return result


class Engine:
    def __init__(self,registry,device='cuda:0'):
        self.registry=registry; self.device=device; self.model=None; self.key=None; self.metadata={}
        self.cache={}  # At most two models; inactive models stay in CPU RAM.
        self.warmed=set()

    def load(self,key):
        if key==self.key: return 0.
        import torch
        start=time.perf_counter()
        if self.model is not None:
            self.model.to('cpu')
        self.model=None; self.key=None; gc.collect(); torch.cuda.empty_cache()
        if key in self.cache:
            model,pipeline=self.cache[key]
            model.to(self.device);model.eval()
            self.model=model;self.pipeline=pipeline;self.key=key
            return time.perf_counter()-start
        from mmengine.config import Config
        from mmcv.transforms import Compose
        from mmyolo.registry import MODELS
        from mmyolo.utils import register_all_modules
        from bdd100k_workflow import load_torch_checkpoint
        entry=self.registry[key]
        if not entry['available']: raise ValueError('Checkpoint/config chưa được attach')
        if not torch.cuda.is_available(): raise ValueError('Demo inference cần GPU Kaggle')
        sha=digest(entry['checkpoint'])
        if entry.get('sha256') and sha!=entry['sha256']: raise ValueError('Checkpoint hash mismatch')
        saved=load_torch_checkpoint(entry['checkpoint']); meta=saved.get('meta',{})
        state={k.removeprefix('module.'):v for k,v in saved['state_dict'].items()}
        sparse=meta.get('sparse_reconstruction',{}); contract=sparse.get('contract',{})
        if sparse:
            if sparse.get('stage')!=entry['stage'] or sparse.get('case_id')!=entry['case_id']:
                raise ValueError('Registry stage/case không khớp checkpoint')
            if contract.get('degradation','low_light')!=entry['degradation']:
                raise ValueError('Checkpoint degradation mismatch')
            if contract.get('model_version',entry['architecture'])!=entry['architecture']:
                raise ValueError('Checkpoint architecture mismatch')
        classes=meta.get('dataset_meta',{}).get('classes')
        if classes and tuple(classes)!=CLASSES[entry['degradation']]: raise ValueError('Checkpoint class order mismatch')
        cfg=Config.fromfile(entry['config'])
        expected='YOLOv10ReconstructBackbone' if entry['architecture']=='v10' else 'YOLOv8ReconstructCSPDarknet'
        if cfg.model.backbone.type!=expected: raise ValueError('Config architecture mismatch')
        if cfg.model.bbox_head.head_module.num_classes!=len(entry['classes']): raise ValueError('Config class count mismatch')
        # Experimental sparse cases only change TRAINING. The inherited prediction
        # path is the native full reconstruction detector, with identical tensors.
        cfg.model=serving_model_config(cfg.model)
        register_all_modules()
        with torch.cuda.device(self.device):
            model=MODELS.build(cfg.model); model.init_weights()
            model.load_state_dict(state,strict=True)
            model.to(self.device).eval()
        model.cfg=cfg; model.dataset_meta={'classes':entry['classes']}
        self.pipeline=Compose(serving_pipeline(cfg.test_dataloader.dataset.pipeline))
        self.model=model; self.key=key
        self.cache[key]=(model,self.pipeline)
        while len(self.cache)>2:
            del self.cache[next(k for k in self.cache if k!=key)]
        self.metadata[key]=dict(checkpoint_sha256=sha,checkpoint_epoch=meta.get('epoch'),
            origin=entry.get('origin','self_trained'),stage=entry['stage'],case_id=entry['case_id'],architecture=entry['architecture'],
            degradation=entry['degradation'],provenance_verified=bool(sparse),
            method=method_info(entry['case_id'],entry['stage']),
            training_target=contract.get('target_epochs'),completed_epochs=entry.get('completed_epochs'),
            run_target_epochs=entry.get('run_target_epochs'),precision='fp32')
        del saved,state
        return time.perf_counter()-start

    def predict(self,key,image,return_reconstruction=False):
        import torch
        from mmdet.apis import inference_detector
        from projects.inference_runtime.prediction import prediction_mode
        load_seconds=self.load(key)
        bgr=np.asarray(image)[:,:,::-1].copy()
        # Includes native preprocessing/postprocessing. CUDA synchronized; network
        # and cold checkpoint loading excluded. Warmup is not a measured request.
        with torch.inference_mode(), prediction_mode(self.model,
                decoder_free=not return_reconstruction,render_reconstruction=return_reconstruction):
            if load_seconds or (key,return_reconstruction) not in self.warmed:
                inference_detector(self.model,bgr,test_pipeline=self.pipeline)
                self.warmed.add((key,return_reconstruction))
            torch.cuda.synchronize(); start=time.perf_counter()
            result=inference_detector(self.model,bgr,test_pipeline=self.pipeline)
            torch.cuda.synchronize(); elapsed=(time.perf_counter()-start)*1000
        pred=result.pred_instances.cpu(); boxes=[]
        for bbox,score,label in zip(pred.bboxes.tolist(),pred.scores.tolist(),pred.labels.tolist()):
            if label>=len(self.registry[key]['classes']): raise ValueError('Invalid predicted class index')
            boxes.append(dict(bbox=bbox,score=score,label=self.registry[key]['classes'][label]))
        reconstruction=None
        if return_reconstruction and hasattr(result,'image') and hasattr(result.image,'output'):
            from PIL import Image
            x=result.image.output.detach().float().cpu().permute(1,2,0).numpy()
            reconstruction=Image.fromarray(np.rint(np.clip(x,0,1)*255).astype('uint8'))
        metadata=dict(self.metadata[key],inference_mode='with_reconstruction' if return_reconstruction else 'detection_only',
                      decoder_executed=bool(return_reconstruction),weights_pruned=False)
        return dict(model_id=key,boxes=boxes,metadata=metadata,
            timing=dict(inference_ms=elapsed,model_load_seconds=load_seconds,
                scope='preprocessing + '+('full model incl. reconstruction image interpolation' if return_reconstruction else 'encoder/neck/head, decoder bypassed')+' + postprocessing; CUDA synchronized; excludes upload, cold load, PNG export and warmup'),
            reconstruction=reconstruction)
