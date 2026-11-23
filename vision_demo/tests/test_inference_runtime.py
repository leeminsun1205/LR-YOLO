"""Real CPU tensor checks for the shared decoder bypass; no OpenMMLab/CUDA."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from projects.inference_runtime.prediction import detection_features, prediction_mode

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('inference_benchmark_test',
    ROOT/'notebooks/selective_reconstruction/inference_benchmark.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


class Backbone(nn.Module):
    layers=('stem','stage1','stage2')
    out_indices=(0,1,2)
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv2d(3,2,3,padding=1)
        self.stage1=nn.Conv2d(2,4,3,stride=2,padding=1)
        self.stage2=nn.Conv2d(4,8,3,stride=2,padding=1)
        self.up1=nn.Conv2d(8,4,1)
        self.up2=nn.Conv2d(4,2,1)
        self.up3=nn.Conv2d(2,3,1)
    def forward(self,inputs):
        features=[]
        for name in self.layers:
            inputs=getattr(self,name)(inputs);features.append(inputs)
        rgb=self.up3(self.up2(self.up1(inputs)))
        return tuple(features),rgb


class Neck(nn.Module):
    def forward(self,features):return tuple(f*2 for f in features)


class Head(nn.Module):
    def forward(self,features):return ([f.mean((2,3)) for f in features],None)
    def predict(self,features,samples,rescale=True):
        value=sum(f.mean() for f in features)
        return [SimpleNamespace(bboxes=value.expand(1,4),scores=value.sigmoid().reshape(1),
                                labels=torch.zeros(1,dtype=torch.long)) for _ in samples]


class Model(nn.Module):
    with_neck=True
    def __init__(self):
        super().__init__();self.backbone=Backbone();self.neck=Neck();self.bbox_head=Head()
    def extract_feat(self,inputs):
        features,rgb=self.backbone(inputs);return self.neck(features),rgb
    def add_pred_to_datasample(self,samples,predictions):
        for sample,prediction in zip(samples,predictions):sample.pred_instances=prediction
        return samples
    def predict(self,inputs,samples,rescale=True):
        features,rgb=self.extract_feat(inputs)
        result=self.add_pred_to_datasample(samples,self.bbox_head.predict(features,samples,rescale))
        for sample in result:sample.image=rgb
        return result


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=Model().eval();self.inputs=torch.randn(1,3,12,12)

    def test_features_heads_and_predictions_match_without_decoder_calls(self):
        calls=[]
        handle=self.model.backbone.up1.register_forward_hook(lambda *args:calls.append(1))
        state={k:v.clone() for k,v in self.model.state_dict().items()}
        with torch.inference_mode():
            reference=detection_features(self.model,self.inputs,False)
            self.assertEqual(len(calls),1);calls.clear()
            fast=detection_features(self.model,self.inputs,True)
            self.assertFalse(calls)
            bench.tensor_tree_close(reference,fast,rtol=0,atol=0)
            bench.tensor_tree_close(self.model.bbox_head(reference),self.model.bbox_head(fast),rtol=0,atol=0)
            outputs=[]
            for skip in (False,True):
                with prediction_mode(self.model,decoder_free=skip):
                    outputs.append(self.model.predict(self.inputs,[SimpleNamespace()])[0])
            torch.testing.assert_close(outputs[0].pred_instances.bboxes,outputs[1].pred_instances.bboxes)
            self.assertFalse(hasattr(outputs[1],'image'))
        handle.remove()
        self.assertEqual(state.keys(),self.model.state_dict().keys())
        for key,value in state.items():torch.testing.assert_close(value,self.model.state_dict()[key],rtol=0,atol=0)
        self.assertIs(self.model.predict.__func__,Model.predict)

    def test_context_restores_native_rgb_mode_after_exception_and_nested_mode(self):
        with self.assertRaisesRegex(RuntimeError,'probe'):
            with prediction_mode(self.model):
                with prediction_mode(self.model,decoder_free=False):pass
                raise RuntimeError('probe')
        self.assertNotIn('predict',self.model.__dict__)
        with torch.inference_mode(),prediction_mode(self.model,decoder_free=False,render_reconstruction=True):
            self.assertTrue(hasattr(self.model.predict(self.inputs,[SimpleNamespace()])[0],'image'))

    def test_rejects_training_and_contradictory_modes(self):
        with self.assertRaises(ValueError):
            with prediction_mode(self.model,decoder_free=True,render_reconstruction=True):pass
        self.model.train()
        with self.assertRaises(RuntimeError):
            with prediction_mode(self.model):pass

    def test_latency_statistics_and_tensor_mismatches_are_visible(self):
        report=bench.summarize([.010,.020,.030])
        self.assertAlmostEqual(report['mean_ms'],20.)
        self.assertAlmostEqual(report['median_ms'],20.)
        self.assertAlmostEqual(report['p95_ms'],29.)
        self.assertAlmostEqual(report['throughput_batch1_fps'],50.)
        with self.assertRaises(ValueError):bench.summarize([0.])
        with self.assertRaises(AssertionError):bench.tensor_tree_close([torch.ones(1)],[],rtol=0,atol=0)

    def test_selection_needs_no_clean_or_prototype_and_rejects_ambiguity(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);checkpoint=path/'best.pth';checkpoint.write_bytes(b'fixture')
            (path/'resolved_config.py').write_text('# trusted fixture')
            record=dict(id='s3',label='S3',architecture='v8',degradation='haze',stage='dark',
                        case_id=9,checkpoint=str(checkpoint),config=str(path/'resolved_config.py'))
            result,missing=bench.select_models([('haze','v8')],[6,9],[record],[])
            self.assertEqual(result,[record]);self.assertEqual(missing,['haze/v8/S1'])
            with self.assertRaises(ValueError):bench.select_models([('haze','v8')],[9],[record,record],[])
            with self.assertRaises(FileNotFoundError):
                bench.select_models([('haze','v8')],[6,9],[record],[],require_all=True)
            result,_=bench.select_models([('haze','v8')],[9],[record,record],[],
                                         explicit={'haze/v8/S3':str(checkpoint)})
            self.assertEqual(result[0]['case_id'],9)

    def test_equivalence_audit_checks_raw_outputs_and_actual_decoder_calls(self):
        from unittest.mock import patch
        batch=dict(inputs=self.inputs,data_samples=[SimpleNamespace()])
        with patch.object(bench,'prepare_batch',return_value=batch):
            report=bench.check_equivalence(self.model,None,[None],count=1)
        self.assertEqual(report['status'],'passed')
        self.assertTrue(report['decoder_bypassed'])
        self.assertTrue(report['raw_head_match'])
        self.assertTrue(report['postprocessed_predictions_match'])
        self.assertNotIn('predict',self.model.__dict__)

    def test_ambiguous_data_fallback_preserves_explicit_errors(self):
        from unittest.mock import Mock
        protocol=SimpleNamespace(prepare_final=Mock(side_effect=[
            RuntimeError('Multiple voc manifests match; set root'),dict(count=2734)]))
        result=bench.prepare_spec(protocol,{},'/cache','voc',family='haze')
        self.assertEqual(result['count'],2734)
        self.assertEqual(protocol.prepare_final.call_args.kwargs['search_roots'],())
        protocol.prepare_final=Mock(side_effect=RuntimeError('Multiple voc manifests match; set root'))
        with self.assertRaises(RuntimeError):
            bench.prepare_spec(protocol,{'voc':dict(root='/explicit')},'/cache','voc',family='haze')
        self.assertEqual(protocol.prepare_final.call_count,1)


if __name__=='__main__':unittest.main()
