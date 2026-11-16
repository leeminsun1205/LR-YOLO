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
        for case in (0,1,3,6):
            r=await self.client.get(f'/api/examples/{self.key}/sampling?case_id={case}&seed=42')
            self.assertEqual(r.status_code,200,r.text[:100] if r.status_code!=200 else '')
            self.assertEqual(decode_image(r.content).size,(640,640))
        self.assertEqual(self.engine.inputs,[])

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

    async def test_registry_and_manifest_reject_conflicts(self):
        d=dict(id='a',architecture='v10',degradation='low_light',stage='clean',case_id=0,checkpoint='/missing/x.pth',config='/missing/cfg.py')
        p=self.root/'models.json';p.write_text(json.dumps([d]));self.assertFalse(load_registry(p)['a']['available'])
        p.write_text(json.dumps([d,d]))
        with self.assertRaises(ValueError):load_registry(p)
        (self.root/'ImageSets/Main/val.txt').write_text('../bad\n')
        with self.assertRaises(ValueError):Gallery([self.spec])

if __name__=='__main__':unittest.main()
