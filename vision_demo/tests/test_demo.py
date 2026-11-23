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
    def __init__(self):self.inputs=[];self.reconstruction_modes=[]
    def predict(self,key,image,return_reconstruction=False):
        self.inputs.append(np.asarray(image).copy())
        self.reconstruction_modes.append(return_reconstruction)
        return dict(model_id=key,boxes=[dict(bbox=[2,3,18,19],label='car',score=.9)],
            metadata=dict(checkpoint_epoch=4,provenance_verified=True),
            timing=dict(inference_ms=1.,model_load_seconds=0.,scope='TEST ONLY'),
            reconstruction=image.copy() if return_reconstruction else None)


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

    async def test_detection_only_default_and_explicit_reconstruction_request(self):
        for request_rgb in (False,True):
            response=await self.client.post('/api/jobs',params=dict(
                model_ids=json.dumps(['a']),return_reconstruction=request_rgb),content=self.bytes)
            self.assertEqual(response.status_code,202)
            for _ in range(100):
                out=(await self.client.get('/api/jobs/'+response.json()['id'])).json()
                if out['status'] in ('completed','failed'):break
                await asyncio.sleep(.01)
            self.assertEqual(out['status'],'completed',out)
            self.assertEqual(self.engine.reconstruction_modes[-1],request_rgb)
            self.assertEqual('reconstruction_url' in out['results'][0],request_rgb)
            self.assertEqual(out['degradation']['return_reconstruction'],request_rgb)

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

    async def test_haze_has_random_preview_comparison_export_and_all_patch_methods(self):
        # Exercise the real API/image/sampler paths with a five-class haze gallery.
        haze=Gallery([dict(self.spec,id='haze',degradation='haze')])
        self.gallery.items.update(haze.items)
        key=next(iter(haze.items))
        for case in (0,6,8,9):
            ident=f'haze-{case}'
            self.registry[ident]=dict(id=ident,label=ident,available=True,
                degradation='haze',architecture='v8',stage='dark',case_id=case,
                classes=list(CLASSES['haze']))
        r=await self.client.get('/api/examples/random?degradation=haze')
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['degradation'],'haze')
        detail=(await self.client.get('/api/examples/'+key)).json()
        self.assertEqual(detail['annotations'][0]['label'],'car')
        preview=await self.client.post('/api/preview',params=dict(example_id=key,mode='haze',beta=.10))
        self.assertEqual(preview.status_code,200)
        expected=np.asarray(decode_image(preview.content))
        for case in (6,8,9):
            illustration=await self.client.get(f'/api/examples/{key}/sampling?case_id={case}&seed=42')
            self.assertEqual(illustration.status_code,200)
            self.assertEqual(decode_image(illustration.content).size,(640,640))
            r=await self.client.post('/api/jobs',params=dict(
                example_id=key,mode='haze',beta=.10,return_reconstruction=True,
                model_ids=json.dumps(['haze-0',f'haze-{case}'])))
            self.assertEqual(r.status_code,202,r.text)
            for _ in range(100):
                out=(await self.client.get('/api/jobs/'+r.json()['id'])).json()
                if out['status'] in ('completed','failed'):break
                await asyncio.sleep(.01)
            self.assertEqual(out['status'],'completed',out)
            self.assertEqual(out['degradation']['mode'],'haze')
            self.assertEqual(out['example']['degradation'],'haze')
            for actual in self.engine.inputs[-2:]:np.testing.assert_array_equal(actual,expected)
            self.assertEqual(out['results'][1]['classes'],list(CLASSES['haze']))
            self.assertEqual(len(out['results'][1]['classes']),5)
            for result in out['results']:
                self.assertEqual((await self.client.get(result['annotated_url'])).status_code,200)
                self.assertEqual((await self.client.get(result['reconstruction_url'])).status_code,200)
            export=(await self.client.get(f'/api/artifacts/{out["id"]}/result.json')).json()
            self.assertEqual(export['degradation']['beta'],.10)
            self.assertNotIn('ap50',export)

    async def test_demo_notebook_enables_real_galleries_and_only_two_original_v8_models(self):
        import ast
        builder=Path(__file__).resolve().parents[2]/'notebooks/demo/build_notebook.py'
        constants=[n.value for n in ast.walk(ast.parse(builder.read_text()))
                   if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        source=next(s for s in constants if 'GALLERY_FAMILIES=' in s)
        assignments={n.targets[0].id:ast.literal_eval(n.value)
                     for n in ast.parse(source).body if isinstance(n,ast.Assign)
                     and isinstance(n.targets[0],ast.Name)
                     and n.targets[0].id in ('GALLERY_FAMILIES','AUTHOR_RELEASES','MODELS',
                                            'DOWNLOAD_REAL_GALLERIES','PRELOAD_MODELS')}
        self.assertEqual(assignments['GALLERY_FAMILIES'],['low_light','haze'])
        self.assertEqual(set(assignments['AUTHOR_RELEASES']),
                         {'voc_dark_v8','voc_haze_v8'})
        self.assertEqual(assignments['MODELS'],[])
        self.assertTrue(assignments['DOWNLOAD_REAL_GALLERIES'])
        self.assertTrue(assignments['PRELOAD_MODELS'])

    async def test_exdark_extensions_aliases_and_separate_random_sources(self):
        (self.root/'ImageSets/Main/real.txt').write_text('000001.jpg\n')
        (self.root/'Annotations/000001.jpg.xml').write_text(
            '<annotation><filename>000001.jpg</filename><object><name>People</name>'
            '<bndbox><xmin>2</xmin><ymin>3</ymin><xmax>18</xmax><ymax>19</ymax></bndbox>'
            '</object><object><name>Table</name></object></annotation>')
        gallery=Gallery([dict(self.spec,id='exdark',manifest='ImageSets/Main/real.txt',
                             pool='real',source='ExDark',split='real_test',already_degraded=True)])
        self.gallery.items.update(gallery.items)
        voc=(await self.client.get('/api/examples/random?degradation=low_light')).json()
        self.assertEqual(voc['id'],self.key)
        real=(await self.client.get('/api/examples/random?degradation=low_light&pool=real')).json()
        self.assertEqual(real['source'],'ExDark')
        self.assertTrue(real['already_degraded'])
        detail=(await self.client.get('/api/examples/'+real['id'])).json()
        self.assertEqual([x['label'] for x in detail['annotations']],['person'])
        preview=await self.client.post('/api/preview',params=dict(example_id=real['id']))
        np.testing.assert_array_equal(np.asarray(decode_image(preview.content)),
                                      np.asarray(decode_image((self.root/'JPEGImages/000001.jpg').read_bytes())))

    async def test_real_gallery_preparation_uses_correct_native_domain(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        from vision_demo.kaggle_setup import prepare_gallery
        prepare=Mock(return_value=dict(root=str(self.root),prefix='',manifest='ImageSets/Main/val.txt'))
        protocol=SimpleNamespace(prepare_final=prepare,prepare_training=Mock())
        with patch.dict(sys.modules,{'selective_protocol_data':protocol}):
            for family,domain,source in [('low_light','exdark','ExDark'),('haze','rtts','RTTS')]:
                result=prepare_gallery(family,self.root,pool='real_test')
                self.assertEqual(result['source'],source)
                self.assertEqual(result['pool'],'real')
                self.assertTrue(result['already_degraded'])
                self.assertEqual(prepare.call_args.args[2],domain)
                self.assertEqual(prepare.call_args.kwargs['degradation'],family)
        protocol.prepare_training.assert_not_called()

    async def test_preload_warms_both_families_before_ready(self):
        from unittest.mock import patch
        self.gallery.items.update(Gallery([dict(self.spec,id='haze',degradation='haze')]).items)
        engine=FakeEngine()
        app=create_app(registry=self.registry,gallery=self.gallery,reports=[],engine=engine,
                       workdir=self.root/'warmup')
        with patch.dict('os.environ',{'LRYOLO_PRELOAD':json.dumps(['h','a'])}):
            async with app.router.lifespan_context(app):
                self.assertEqual(len(engine.inputs),2)

    async def test_cached_model_switch_offloads_gpu_before_activation(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from vision_demo.server.inference import Engine
        events=[]
        class Module:
            def __init__(self,name):self.name=name
            def to(self,device):events.append((self.name,device));return self
            def eval(self):return None  # Support native modules with nonchainable train/eval.
        old,new=Module('old'),Module('new')
        engine=Engine({});engine.model=old;engine.key='old'
        engine.cache={'old':(old,'old-pipeline'),'new':(new,'new-pipeline')}
        torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None))
        with patch.dict(sys.modules,{'torch':torch}):engine.load('new')
        self.assertEqual(events,[('old','cpu'),('new','cuda:0')])
        self.assertIs(engine.model,new)
        self.assertEqual(engine.pipeline,'new-pipeline')

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
