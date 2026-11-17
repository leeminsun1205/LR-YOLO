"""Trusted native checkpoint inference; lazy imports, one cached model, one GPU."""
import gc
import copy
import time
from pathlib import Path
import numpy as np
from .core import CLASSES, digest


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

    def load(self,key):
        if key==self.key: return 0.
        import torch
        from mmengine.config import Config
        from mmcv.transforms import Compose
        from mmyolo.registry import MODELS
        from mmyolo.utils import register_all_modules
        from bdd100k_workflow import load_torch_checkpoint
        start=time.perf_counter(); entry=self.registry[key]
        if not entry['available']: raise ValueError('Checkpoint/config chưa được attach')
        if not torch.cuda.is_available(): raise ValueError('Demo inference cần GPU Kaggle')
        self.model=None; self.key=None; gc.collect(); torch.cuda.empty_cache()
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
        if cfg.model.type=='SelectiveReconstructionDetector':
            cfg.model.type='YOLOReconstructDehazeDetector'
            for p in ('sparse_case','crop_count','crop_core','crop_halo','sampling_seed','coverage_candidates'): cfg.model.pop(p,None)
        if cfg.model.type not in ('YOLOReconstructDetector','YOLOReconstructDehazeDetector'):
            raise ValueError('Unsupported detector type; use native/selective reconstruction checkpoints')
        cfg.model.data_preprocessor.type='YOLOv5DetDataPreprocessor'
        cfg.model.pop('init_cfg',None); cfg.model.backbone.pop('init_cfg',None)
        register_all_modules()
        with torch.cuda.device(self.device):
            model=MODELS.build(cfg.model); model.init_weights()
            model.load_state_dict(state,strict=True)
            model.to(self.device).eval()
        model.cfg=cfg; model.dataset_meta={'classes':entry['classes']}
        self.pipeline=Compose(serving_pipeline(cfg.test_dataloader.dataset.pipeline))
        self.model=model; self.key=key
        self.metadata[key]=dict(checkpoint_sha256=sha,checkpoint_epoch=meta.get('epoch'),
            origin=entry.get('origin','self_trained'),stage=entry['stage'],case_id=entry['case_id'],architecture=entry['architecture'],
            degradation=entry['degradation'],provenance_verified=bool(sparse),
            training_target=contract.get('target_epochs'),completed_epochs=entry.get('completed_epochs'),
            run_target_epochs=entry.get('run_target_epochs'),precision='fp32')
        del saved,state
        return time.perf_counter()-start

    def predict(self,key,image):
        import torch
        from mmdet.apis import inference_detector
        load_seconds=self.load(key)
        bgr=np.asarray(image)[:,:,::-1].copy()
        # Includes native preprocessing/postprocessing. CUDA synchronized; network
        # and cold checkpoint loading excluded. Warmup is not a measured request.
        with torch.inference_mode():
            if load_seconds: inference_detector(self.model,bgr,test_pipeline=self.pipeline)
            torch.cuda.synchronize(); start=time.perf_counter()
            result=inference_detector(self.model,bgr,test_pipeline=self.pipeline)
            torch.cuda.synchronize(); elapsed=(time.perf_counter()-start)*1000
        pred=result.pred_instances.cpu(); boxes=[]
        for bbox,score,label in zip(pred.bboxes.tolist(),pred.scores.tolist(),pred.labels.tolist()):
            if label>=len(self.registry[key]['classes']): raise ValueError('Invalid predicted class index')
            boxes.append(dict(bbox=bbox,score=score,label=self.registry[key]['classes'][label]))
        reconstruction=None
        if hasattr(result,'image') and hasattr(result.image,'output'):
            from PIL import Image
            x=result.image.output.detach().float().cpu().permute(1,2,0).numpy()
            reconstruction=Image.fromarray(np.rint(np.clip(x,0,1)*255).astype('uint8'))
        return dict(model_id=key,boxes=boxes,metadata=self.metadata[key],
            timing=dict(inference_ms=elapsed,model_load_seconds=load_seconds,
                scope='native preprocessing + full model prediction + postprocessing; CUDA synchronized; excludes upload, cold load and warmup'),
            reconstruction=reconstruction)
