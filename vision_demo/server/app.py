"""Run: uvicorn vision_demo.server.app:app --workers 1 (registry paths via environment)."""
import io
from contextlib import asynccontextmanager
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from .core import Gallery, load_registry, load_reports, digest
from .imaging import MAX_BYTES, decode_image, degrade, png, annotated
from .inference import Engine
from ..methods import METHODS, method_info


def create_app(registry=None,gallery=None,reports=None,engine=None,workdir=None,webdir=None):
    base=Path(__file__).resolve().parents[1]
    registry=registry if registry is not None else load_registry(os.getenv('LRYOLO_MODELS',str(base/'config/models.example.json')))
    gallery=gallery if gallery is not None else Gallery(json.loads(Path(os.getenv('LRYOLO_GALLERY',str(base/'config/gallery.example.json'))).read_text()))
    reports=reports if reports is not None else load_reports(os.getenv('LRYOLO_REPORTS',str(base/'config/reports.example.json')))
    engine=engine or Engine(registry)
    work=Path(workdir or os.getenv('LRYOLO_OUTPUT','/tmp/lryolo_demo_artifacts'));work.mkdir(parents=True,exist_ok=True)
    web=Path(webdir or base/'web/dist')
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='inference')
    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=False,cancel_futures=True)
    app=FastAPI(title='LR-YOLO Vision Lab',docs_url='/api/docs',lifespan=lifespan)
    jobs={}; lock=threading.Lock()

    async def image_data(request,example_id):
        chunks=[]; size=0
        async for chunk in request.stream():
            size+=len(chunk)
            if size>MAX_BYTES: raise ValueError('Ảnh phải nhỏ hơn20MB')
            chunks.append(chunk)
        if example_id:
            if size: raise ValueError('Chọn upload hoặc ảnh mẫu, không chọn cả hai')
            if example_id not in gallery.items: raise ValueError('Không tìm thấy ảnh mẫu')
            return gallery.items[example_id]['path'].read_bytes()
        if not size: raise ValueError('Hãy chọn một ảnh')
        return b''.join(chunks)

    def settings(mode,gamma,beta,family=None):
        if mode not in ('none','low_light','haze'): raise ValueError('Invalid degradation')
        if family and mode not in ('none',family): raise ValueError('Degradation không khớp model')
        return dict(mode=mode,gamma=gamma,beta=beta,
            scope='offline demo on original RGB before native letterbox; quantized PNG; not training float pipeline',
            haze_depth_canvas=640 if mode=='haze' else None)

    @app.get('/api/health')
    def health():
        with lock: pending=sum(j['status'] in ('queued','loading','running') for j in jobs.values())
        return dict(status='ready',available_models=sum(e['available'] for e in registry.values()),
                    gallery_images=len(gallery.items),pending_jobs=pending,gpu_worker_count=1)

    @app.get('/api/models')
    def models():
        return [{k:v for k,v in e.items() if k not in ('checkpoint','config')} for e in registry.values()]

    @app.get('/api/experiments')
    def experiments(): return reports

    @app.get('/api/methods')
    def methods(): return [method_info(case) for case in METHODS]

    @app.get('/api/examples/random')
    def random_example(degradation:str='low_light'):
        try: return gallery.choose(degradation)
        except ValueError as exc: raise HTTPException(404,str(exc))

    @app.get('/api/examples/{key}')
    def example(key:str):
        if key not in gallery.items: raise HTTPException(404,'Không tìm thấy ảnh mẫu')
        return dict(**gallery.public(gallery.items[key]),annotations=gallery.annotations(key),
            annotation_coordinates='raw VOC XML; no benchmark metric computed')

    @app.get('/api/examples/{key}/image')
    def example_image(key:str):
        if key not in gallery.items: raise HTTPException(404,'Không tìm thấy ảnh mẫu')
        return FileResponse(gallery.items[key]['path'],media_type='image/jpeg')

    @app.get('/api/examples/{key}/sampling')
    def sampling_view(key:str,case_id:int=6,seed:int=42):
        if key not in gallery.items: raise HTTPException(404,'Không tìm thấy ảnh VOC')
        if case_id not in (0,1,3,6,8,9): raise HTTPException(422,'Minh họa LR-YOLO gốc / S1 / S2 / S3')
        import importlib.util
        from PIL import Image,ImageDraw
        item=gallery.items[key];im=decode_image(item['path'].read_bytes())
        ratio=640/max(im.size);w,h=round(im.width*ratio),round(im.height*ratio)
        x,y=(640-w)//2,(640-h)//2
        canvas=Image.new('RGB',(640,640),(114,114,114));canvas.paste(im.resize((w,h)),(x,y))
        draw=ImageDraw.Draw(canvas);boxes=[]
        for obj in gallery.annotations(key):
            a,b,c,d=obj['bbox'];box=[a*ratio+x,b*ratio+y,c*ratio+x,d*ratio+y]
            boxes.append(box);draw.rectangle(box,outline='#60a5fa',width=1)
        if case_id in (0,1):draw.rectangle([1,1,638,638],outline='#2dd4bf',width=5)
        else:
            repo=Path(os.getenv('LRYOLO_REPO',str(base.parent)))
            path=repo/'projects/sparse_reconstruction/sampling.py'
            if not path.is_file():raise HTTPException(503,'Sampling source missing from demo checkout')
            spec=importlib.util.spec_from_file_location('demo_sampler',path)
            sampler=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampler)
            if case_id in (8,9):regions=sampler.sample_coverage_regions(640,640,boxes,seed=seed)
            else:regions=sampler.sample_regions(640,640,boxes,seed=seed,object_probability=0,
                stratified_pair=case_id==6)
            for r in regions:
                color='#fbbf24' if r['guided'] else '#2dd4bf'
                draw.rectangle([r['x'],r['y'],r['x']+r['size'],r['y']+r['size']],outline=color,width=2)
                left,top=r['x']+r['core_x'],r['y']+r['core_y']
                draw.rectangle([left,top,left+r['core'],top+r['core']],outline=color,width=4)
        draw.rectangle([0,0,640,28],fill='#10222e')
        name='LR-YOLO' if case_id==0 else method_info(case_id)['name']
        draw.text((10,8),f'{name} / seed {seed} / training regions, not inference attention',fill='#e0fff7')
        return Response(png(canvas),media_type='image/png',headers={
            'X-LRYOLO-Case':str(case_id),
            'X-LRYOLO-Sampling-Case':str(8 if case_id==9 else case_id)})

    @app.post('/api/preview')
    async def preview(request:Request,example_id:str='',mode:str='none',
                      gamma:float=2.5,beta:float=.08):
        try:
            data=await image_data(request,example_id); source=decode_image(data)
            settings(mode,gamma,beta)
            return Response(png(degrade(source,mode,gamma,beta)),media_type='image/png')
        except (ValueError,OSError) as exc: raise HTTPException(422,str(exc))

    def perform(ident,data,keys,options,example_id):
        job=jobs[ident]; folder=work/ident; folder.mkdir()
        try:
            original=decode_image(data)
            processed=degrade(original,options['mode'],options['gamma'],options['beta'])
            (folder/'original.png').write_bytes(png(original));(folder/'input.png').write_bytes(png(processed))
            job.update(status='loading',progress='Đang tải checkpoint',started=time.time())
            results=[]
            for key in keys:
                job['progress']=f'Đang chạy {registry[key]["label"]}'
                result=engine.predict(key,processed)
                job['status']='running'
                restoration=result.pop('reconstruction',None)
                if restoration is not None:
                    name=f'{key}_reconstruction.png';(folder/name).write_bytes(png(restoration))
                    result['reconstruction_url']=f'/api/artifacts/{ident}/{name}'
                result['classes']=registry[key]['classes'];result['label']=registry[key]['label']
                name=f'{key}_annotated.png'
                (folder/name).write_bytes(png(annotated(processed,result['boxes'],classes=result['classes'])))
                result['annotated_url']=f'/api/artifacts/{ident}/{name}'
                results.append(result)
            output=dict(id=ident,status='completed',input_url=f'/api/artifacts/{ident}/input.png',
                original_url=f'/api/artifacts/{ident}/original.png',input_sha256=digest(folder/'input.png'),
                width=processed.width,height=processed.height,degradation=options,results=results,
                example=gallery.public(gallery.items[example_id]) if example_id else None,
                elapsed_seconds=time.time()-job['created'],
                note='Detection counts/confidences are not AP; inference unchanged across training cases.')
            (folder/'result.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
            job.update(output);job['progress']='Hoàn tất'
        except Exception as exc:
            job.update(status='failed',error=f'{type(exc).__name__}: {exc}',progress='Không xử lý được ảnh')

    @app.post('/api/jobs',status_code=202)
    async def submit(request:Request,example_id:str='',model_ids:str='[]',
                     mode:str='none',gamma:float=2.5,beta:float=.08):
        try:
            keys=json.loads(model_ids)
            if not isinstance(keys,list) or not 1<=len(keys)<=2 or len(set(keys))!=len(keys):
                raise ValueError('Chọn một hoặc hai model khác nhau')
            if any(not isinstance(k,str) or k not in registry or not registry[k]['available'] for k in keys):
                raise ValueError('Model chưa được attach hoặc không hợp lệ')
            families={registry[k]['degradation'] for k in keys}
            if len(families)!=1: raise ValueError('So sánh cần cùng degradation/class semantics')
            options=settings(mode,gamma,beta,next(iter(families)))
            data=await image_data(request,example_id)
            # Validate image/settings before accepting a GPU job.
            im=decode_image(data);degrade(im,mode,gamma,beta)
            with lock:
                if len(jobs)>=100: raise HTTPException(429,'Phiên demo đã đủ100lượt; hãy export và khởi động lại server')
                if sum(j['status'] in ('queued','loading','running') for j in jobs.values())>=4:
                    raise HTTPException(429,'GPU đang bận. Vui lòng chờ lượt hiện tại')
                ident=uuid.uuid4().hex
                jobs[ident]=dict(id=ident,status='queued',progress='Đang chờ GPU',created=time.time())
            pool.submit(perform,ident,data,keys,options,example_id)
            return dict(id=ident,status='queued')
        except (ValueError,TypeError,OSError) as exc: raise HTTPException(422,str(exc))

    @app.get('/api/jobs/{ident}')
    def job(ident:str):
        if ident not in jobs: raise HTTPException(404,'Job không tồn tại trong phiên này')
        return dict(jobs[ident])

    @app.get('/api/artifacts/{ident}/{name}')
    def artifact(ident:str,name:str):
        if not re.fullmatch('[a-f0-9]{32}',ident) or ident not in jobs: raise HTTPException(404)
        allowed={'input.png','original.png','result.json'}
        allowed.update(f'{k}_{suffix}.png' for k in registry for suffix in ('reconstruction','annotated'))
        if name not in allowed or not (work/ident/name).is_file(): raise HTTPException(404)
        return FileResponse(work/ident/name)

    if (web/'assets').is_dir(): app.mount('/assets',StaticFiles(directory=web/'assets'),name='assets')

    @app.get('/')
    def index():
        if not (web/'index.html').is_file(): raise HTTPException(503,'Chưa build frontend: npm ci && npm run build')
        return FileResponse(web/'index.html')

    return app


app=create_app()
