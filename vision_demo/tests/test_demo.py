"""Real ASGI/image tests, with explicit fake inference (no Torch/GPU required)."""
import io
import asyncio
import httpx
import json
import tempfile
import time
import unittest
from pathlib import Path
from PIL import Image
import numpy as np
from vision_demo.server.core import Gallery, load_registry, load_reports, CLASSES
from vision_demo.server.imaging import decode_image,degrade,png
from vision_demo.server.app import create_app


class FakeEngine:
    """Deterministic test double; NEVER wired into shipped serving app."""
    def __init__(self):self.inputs=[]
    def predict(self,key,image):
        self.inputs.append(np.asarray(image).copy())
        return dict(model_id=key,boxes=[dict(bbox=[2,3,18,19],label='car',score=.9)],
            metadata=dict(checkpoint_epoch=4,provenance_verified=True),
            timing=dict(inference_ms=1.,model_load_seconds=0.,scope='TEST ONLY'),
            reconstruction=image.copy())


class DemoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for folder in ('JPEGImages','Annotations','ImageSets/Main'):(self.root/folder).mkdir(parents=True)
        im=Image.new('RGB',(64,32),(150,100,50));self.bytes=png(im)
        im.save(self.root/'JPEGImages/000001.jpg')
        (self.root/'ImageSets/Main/val.txt').write_text('000001\n')
        (self.root/'Annotations/000001.xml').write_text('<annotation><object><name>car</name><bndbox><xmin>2</xmin><ymin>3</ymin><xmax>18</xmax><ymax>19</ymax></bndbox></object><object><name>sofa</name></object></annotation>')
        self.spec=dict(id='voc',root=str(self.root),manifest='ImageSets/Main/val.txt',prefix='',degradation='low_light',split='clean_val',source='fixture')
        self.gallery=Gallery([self.spec]);self.key=next(iter(self.gallery.items))
        self.registry={k:dict(id=k,label=k,available=True,degradation=family,classes=list(CLASSES[family])) for k,family in [('a','low_light'),('b','low_light'),('h','haze')]}
        self.engine=FakeEngine();self.app=create_app(registry=self.registry,gallery=self.gallery,reports=[],engine=self.engine,workdir=self.root/'artifacts')
        self.lifecycle=self.app.router.lifespan_context(self.app)
        await self.lifecycle.__aenter__()
        self.client=httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),base_url='http://test')
        async def heartbeat():
            while True:await asyncio.sleep(.01)
        self.heartbeat=asyncio.create_task(heartbeat())

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.lifecycle.__aexit__(None,None,None)
        self.heartbeat.cancel()
        self.tmp.cleanup()

    async def test_gallery_random_annotation_and_no_paths_exposed(self):
        r=await self.client.get('/api/examples/random?degradation=low_light');self.assertEqual(r.status_code,200)
        self.assertNotIn('path',r.json());self.assertNotIn('xml',r.json())
        detail=(await self.client.get('/api/examples/'+self.key)).json();self.assertEqual(len(detail['annotations']),1)
        self.assertEqual(detail['annotations'][0]['label'],'car')
        self.assertEqual((await self.client.get('/api/examples/random?degradation=haze')).status_code,404)

    async def test_gamma_and_haze_are_explicit_finite_and_preserve_geometry(self):
        source=decode_image(self.bytes);dark=degrade(source,'low_light',2.5)
        self.assertEqual(dark.size,source.size)
        self.assertTrue(np.all(np.asarray(dark)<=np.asarray(source)))
        self.assertEqual(png(degrade(source,'none')),png(source))
        self.assertTrue(np.isfinite(np.asarray(degrade(source,'haze'))).all())
        for value in (float('nan'),float('inf'),1.,6.):
            with self.assertRaises(ValueError):degrade(source,'low_light',gamma=value)
        with self.assertRaises(ValueError):decode_image(b'not an image')

    async def test_preview_never_runs_model(self):
        r=await self.client.post('/api/preview?mode=low_light&gamma=3',content=self.bytes)
        self.assertEqual(r.status_code,200);self.assertEqual(decode_image(r.content).size,(64,32))
        self.assertEqual(self.engine.inputs,[])
        self.assertEqual((await self.client.post('/api/preview',content=b'bad')).status_code,422)

    async def test_ab_uses_identical_degraded_input_and_exports_truthful_result(self):
        r=await self.client.post('/api/jobs',params=dict(model_ids=json.dumps(['a','b']),mode='low_light',gamma=3),content=self.bytes)
        self.assertEqual(r.status_code,202);ident=r.json()['id']
        for _ in range(100):
            out=(await self.client.get('/api/jobs/'+ident)).json()
            if out['status'] in ('completed','failed'):break
            await asyncio.sleep(.01)
        self.assertEqual(out['status'],'completed',out)
        np.testing.assert_array_equal(*self.engine.inputs)
        self.assertEqual(len(out['results']),2)
        self.assertEqual((await self.client.get(out['results'][0]['annotated_url'])).status_code,200)
        export=(await self.client.get(f'/api/artifacts/{ident}/result.json')).json()
        self.assertEqual(export['input_sha256'],out['input_sha256'])
        self.assertNotIn('ap50',out)
        self.assertEqual((await self.client.get(f'/api/artifacts/{ident}/secret.txt')).status_code,404)

    async def test_invalid_comparisons_and_degradation_do_not_queue_gpu(self):
        for keys,mode in [(['a','h'],'none'),(['a','a'],'none'),(['a'],'haze'),(['unknown'],'none')]:
            r=await self.client.post('/api/jobs',params=dict(model_ids=json.dumps(keys),mode=mode),content=self.bytes)
            self.assertEqual(r.status_code,422,r.text)
        self.assertEqual(self.engine.inputs,[])

    async def test_gallery_jobs_keep_original_by_default_and_block_dual_source(self):
        r=await self.client.post('/api/preview',params=dict(example_id=self.key),content=self.bytes)
        self.assertEqual(r.status_code,422)
        r=await self.client.post('/api/preview',params=dict(example_id=self.key))
        self.assertEqual(r.status_code,200)
        raw=decode_image((self.root/'JPEGImages/000001.jpg').read_bytes())
        np.testing.assert_array_equal(np.asarray(raw),np.asarray(decode_image(r.content)))

    async def test_sampling_is_real_annotation_based_not_inference(self):
        for case in (0,1,3,6,8,9):
            r=await self.client.get(f'/api/examples/{self.key}/sampling?case_id={case}&seed=42')
            self.assertEqual(r.status_code,200,r.text[:100] if r.status_code!=200 else '')
            self.assertEqual(decode_image(r.content).size,(640,640))
        self.assertEqual(self.engine.inputs,[])

    async def test_current_method_names_and_s3_preserves_s2_regions(self):
        methods=(await self.client.get('/api/methods')).json()
        self.assertEqual([(m['case_id'],m['name']) for m in methods],
                         [(0,'LR-YOLO gốc'),(6,'S1'),(8,'S2'),(9,'S3')])
        images=[]
        for case in (8,9):
            r=await self.client.get(f'/api/examples/{self.key}/sampling?case_id={case}&seed=42')
            self.assertEqual(r.headers['X-LRYOLO-Sampling-Case'],'8')
            images.append(np.asarray(decode_image(r.content))[28:])
        np.testing.assert_array_equal(*images)
        self.assertEqual(self.engine.inputs,[])

    async def test_all_current_methods_serve_native_tensors_without_bank(self):
        from vision_demo.server.inference import serving_model_config
        for version in ('v8','v10'):
            for case in (0,6,8,9):
                original=dict(type='PrototypeMemorySelectiveDetector' if case==9 else
                              'SelectiveReconstructionDetector' if case else 'YOLOReconstructDehazeDetector',
                    backbone=dict(type='YOLOv10ReconstructBackbone' if version=='v10' else 'YOLOv8ReconstructCSPDarknet',init_cfg={'type':'Pretrained'}),
                    data_preprocessor=dict(type='YOLOv5HazeDetDataPreprocessor',std=[255,255,255]),
                    bbox_head=dict(head_module=dict(num_classes=5)))
                if case:
                    original.update(sparse_case=8 if case==9 else case,crop_count=2,
                                    crop_core=192,crop_halo=32,sampling_seed=42,coverage_candidates=8)
                if case==9:
                    original.update(prototype_directory='/missing/training-only-bank',
                        prototype_identity='identity',prototype_bank_sha256='hash',prototype_weight=1.,
                        prototype_warmup_updates=100,prototype_cap=8,prototype_profile=False)
                before=json.loads(json.dumps(original))
                result=serving_model_config(original)
                self.assertEqual(original,before)
                self.assertEqual(result['type'],'YOLOReconstructDehazeDetector')
                self.assertEqual(result['data_preprocessor']['type'],'YOLOv5DetDataPreprocessor')
                self.assertEqual(result['bbox_head'],original['bbox_head'])
                self.assertEqual(result['data_preprocessor']['std'],[255,255,255])
                self.assertFalse(any(k.startswith('prototype_') for k in result))
                self.assertNotIn('sparse_case',result)
        with self.assertRaisesRegex(ValueError,'coverage'):
            serving_model_config(dict(type='PrototypeMemorySelectiveDetector',sparse_case=6))

    async def test_discovery_keeps_runtime_cases_and_does_not_read_checkpoints(self):
        from vision_demo.kaggle_setup import discover_models
        for case,name in [(6,'S1'),(8,'S2'),(9,'S3')]:
            folder=self.root/f'case{case}';folder.mkdir()
            (folder/'provenance.json').write_text(json.dumps(dict(contract=dict(
                action='train',stage='dark',case_id=case,model_version='v8',degradation='haze'))))
            (folder/'resolved_config.py').write_text('# fixture config')
            (folder/'best_voc_standard_mAP_epoch_100.pth').write_bytes(b'NOT A TORCH CHECKPOINT')
            (folder/'session_state.json').write_text(json.dumps(dict(completed_epochs=100,target_epochs=100)))
        discovered=discover_models([self.root])
        p=self.root/'models.json';p.write_text(json.dumps(discovered))
        registry=load_registry(p)
        self.assertEqual(len(registry),3)
        for model in registry.values():
            self.assertTrue(model['available'])
            self.assertEqual(model['method']['name'],{6:'S1',8:'S2',9:'S3'}[model['case_id']])
            self.assertIn(model['method']['name'],model['label'])

    async def test_bundled_results_have_24_distinct_evaluations_and_no_invented_cost(self):
        rows=load_reports(Path(__file__).resolve().parents[1]/'config/research_results.json')
        self.assertEqual(len(rows),24)
        for family in ('low_light','haze'):
            self.assertEqual(len([r for r in rows if r['degradation']==family]),12)
        for row in rows:
            self.assertIsNone(row['epochs'])
            self.assertNotIn('train_hours',row)
            self.assertNotIn('budget_group',row)
            self.assertTrue(row['source_file'])

    async def test_reports_do_not_accept_percent_as_fraction_or_nan(self):
        p=self.root/'reports.json'
        row=dict(id='r',label='measured',source='self-trained',degradation='low_light',
                 architecture='v10',dataset='VOC_Dark_Val',metric='AP50',epochs=20)
        p.write_text(json.dumps([dict(row,ap50=.7)]));self.assertEqual(load_reports(p)[0]['ap50'],.7)
        for value in (70.,float('nan'),-.1):
            p.write_text(json.dumps([dict(row,ap50=value)]))
            with self.assertRaises(ValueError):load_reports(p)

    async def test_serving_pipeline_never_requires_xml_for_upload(self):
        from vision_demo.server.inference import serving_pipeline
        original=[dict(type='LoadImageFromFile',backend_args=None),
                  dict(type='YOLOv5KeepRatioResize',scale=(640,640)),
                  dict(type='LetterResize',scale=(640,640),pad_val=dict(img=114)),
                  dict(type='LoadAnnotations'),dict(type='mmdet.PackDetInputs',meta_keys=['pad_param'])]
        result=serving_pipeline(original)
        self.assertEqual(result[0]['type'],'mmdet.LoadImageFromNDArray')
        self.assertEqual(len(result),4)
        self.assertEqual(result[1:3],original[1:3])
        self.assertEqual(original[0]['type'],'LoadImageFromFile')

    async def test_ngrok_secret_failure_is_actionable_and_never_starts_tunnel(self):
        from vision_demo.kaggle_setup import configure_ngrok
        from unittest.mock import Mock,patch
        client=Mock();client.get_secret.side_effect=ConnectionError('HTTP404')
        ngrok=Mock()
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'Kaggle Secrets'):
                configure_ngrok(ngrok,client)
            ngrok.set_auth_token.assert_not_called()
            configure_ngrok(ngrok,client,token=' fixture-token ')
            ngrok.set_auth_token.assert_called_once_with('fixture-token')
        client.get_secret.assert_called_once_with('NGROK_AUTH_TOKEN')

    async def test_registry_and_manifest_reject_conflicts(self):
        d=dict(id='a',architecture='v10',degradation='low_light',stage='clean',case_id=0,checkpoint='/missing/x.pth',config='/missing/cfg.py')
        p=self.root/'models.json';p.write_text(json.dumps([d]));self.assertFalse(load_registry(p)['a']['available'])
        p.write_text(json.dumps([d,d]))
        with self.assertRaises(ValueError):load_registry(p)
        (self.root/'ImageSets/Main/val.txt').write_text('../bad\n')
        with self.assertRaises(ValueError):Gallery([self.spec])

if __name__=='__main__':unittest.main()
