"""Build standalone Kaggle host notebook, embedding exact demo files. No execution."""
import base64
import hashlib
import json
import sys
import zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
import textwrap


def cell(kind,source):
    result=dict(cell_type=kind,metadata={},source=textwrap.dedent(source).strip()+'\n')
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result


def main():
    files={}
    for p in (ROOT/'vision_demo').rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('node_modules','__pycache__','tests')):continue
        if p.suffix not in ('.py','.json','.html','.tsx','.ts','.js','.css','.txt'):continue
        files[str(p.relative_to(ROOT))]=p.read_text()
    for name in ('bdd100k_workflow.py','voc_adapter_workflow.py','original_voc_workflow.py'):
        files[name]=(ROOT/'notebooks/demo/support'/name).read_text()
    files['selective_protocol_data.py']=(ROOT/'notebooks/demo/support/selective_protocol_data.py').read_text()
    for name in ('bdd100k','lowlight_adapter','original_runtime','sparse_reconstruction'):
        for p in (ROOT/'projects'/name).glob('*.py'):files[str(p.relative_to(ROOT))]=p.read_text()
    data=json.dumps(files,sort_keys=True).encode();sha=hashlib.sha256(data).hexdigest();payload=base64.b64encode(zlib.compress(data)).decode()
    unpack=f'''
import base64,hashlib,json,sys,zlib
from pathlib import Path
raw=zlib.decompress(base64.b64decode({payload!r}))
assert hashlib.sha256(raw).hexdigest()=={sha!r}
BUNDLE=Path('/kaggle/working/lryolo_demo_{sha[:12]}')
for name,content in json.loads(raw).items():
    rel=Path(name)
    assert not rel.is_absolute() and '..' not in rel.parts
    dest=BUNDLE/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.read_text()!=content:raise FileExistsError(dest)
    dest.write_text(content)
sys.path.insert(0,str(BUNDLE))
print('Demo bundle:',BUNDLE,flush=True)
'''
    cells=[cell('markdown','''
# LR-YOLO Vision Lab · Kaggle demo
React + FastAPI + Ngrok. No OpenAI API or training. Attach checkpoints WITH
resolved_config.py, provenance.json and session_state.json; attach tested MMCV
T4 wheel. Enable GPU and Internet. Create NGROK_AUTH_TOKEN in Kaggle Secrets.
Default gallery downloads the author clean VOC bundle if not mounted. Low-light
and haze use DIFFERENT class lists. Random samples are illustration, not benchmark.
Use a separate session from long training. Read docs/vision_demo_runbook.md locally.
'''),cell('code',unpack),cell('code','''
import os,time,subprocess,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
REPO=Path('/kaggle/working/LR-YOLO-demo')
OUTPUT=Path('/kaggle/working/lryolo_demo_runtime');OUTPUT.mkdir(exist_ok=True)
INSTALL_ENVIRONMENT=True  # Uses known Python3.10.10/Torch2.5.1/MMCV wheel recipe.
MMCV_WHEEL=None  # Set explicit wheel path if several are attached.
GALLERY_FAMILIES=['low_light']  # Add 'haze' to prepare the second native VOC bundle.
GALLERY_POOL='clean_val'  # Or fixed_test: already degraded, do not darken again.
AUTO_DOWNLOAD_DATASETS=True
DATA_CACHE=Path('/kaggle/working/lryolo_voc_data')
DATA_OVERRIDES={}  # Optional family -> explicit native train/clean_val settings.
MODELS=None  # None discovers mounted main-run sidecars; or supply explicit list below.
# MODELS=[dict(id='my-s6',label='YOLOv10 S6',architecture='v10',degradation='low_light',
#   stage='dark',case_id=6,checkpoint='/kaggle/input/.../best_....pth',
#   config='/kaggle/input/.../resolved_config.py')]
REPORTS=[]  # Optional measured rows. See vision_demo/config/reports.schema.example.md locally.
PORT=8501
'''),cell('code','''
import bdd100k_workflow as shared
REPO=shared.bootstrap(REPO,BUNDLE,install_environment=INSTALL_ENVIRONMENT,mmcv_wheel=MMCV_WHEEL,num_gpus=1)
# Serving-only packages, no Torch/NumPy/MMCV upgrades.
from importlib.metadata import version,PackageNotFoundError
required={'fastapi':'fastapi==0.115.6','uvicorn':'uvicorn==0.34.0','pyngrok':'pyngrok>=7,<8'}
missing=[]
for name,spec in required.items():
    try:version(name)
    except PackageNotFoundError:missing.append(spec)
if missing:subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
print('Serving packages ready',flush=True)
'''),cell('code','''
from vision_demo.kaggle_setup import discover_models,prepare_gallery
if MODELS is None:MODELS=discover_models()
print('Model inventory:',json.dumps(MODELS,indent=2),flush=True)
if not MODELS:print('No model found: set MODELS explicitly. Gallery/preview will still work.')
gallery=[]
for family in GALLERY_FAMILIES:
    print('Preparing demo VOC pool:',family,GALLERY_POOL,flush=True)
    gallery.append(prepare_gallery(family,DATA_CACHE,auto_download=AUTO_DOWNLOAD_DATASETS,
        overrides=DATA_OVERRIDES.get(family),pool=GALLERY_POOL))
for name,value in [('models',MODELS),('gallery',gallery),('reports',REPORTS)]:
    (OUTPUT/f'{name}.json').write_text(json.dumps(value,indent=2))
# Frontend is prebuilt in the embedded bundle; Node is not required on Kaggle.
assert (BUNDLE/'vision_demo/web/dist/index.html').is_file(),'Frontend missing; rebuild notebook after npm build.'
print('Registry/gallery configured. Start server in next cell.',flush=True)
'''),cell('code','''
import urllib.request
from kaggle_secrets import UserSecretsClient
from pyngrok import ngrok
# Token stays in memory; never print it or place it in frontend files.
ngrok.set_auth_token(UserSecretsClient().get_secret('NGROK_AUTH_TOKEN'))
if 'api_process' in globals() and api_process.poll() is None:
    raise RuntimeError('Demo already running; stop it using the final cell first.')
env=os.environ.copy()
env.update(LRYOLO_REPO=str(REPO),LRYOLO_MODELS=str(OUTPUT/'models.json'),LRYOLO_GALLERY=str(OUTPUT/'gallery.json'),
    LRYOLO_REPORTS=str(OUTPUT/'reports.json'),LRYOLO_OUTPUT=str(OUTPUT/'artifacts'),
    PYTHONPATH=str(REPO)+os.pathsep+str(BUNDLE)+os.pathsep+env.get('PYTHONPATH',''))
log_path=OUTPUT/'server.log';log_file=log_path.open('a')
api_process=subprocess.Popen([sys.executable,'-m','uvicorn','vision_demo.server.app:app',
    '--host','0.0.0.0','--port',str(PORT),'--workers','1'],cwd=BUNDLE,env=env,
    stdout=log_file,stderr=subprocess.STDOUT)
try:
    for attempt in range(90):
        if api_process.poll() is not None:
            raise RuntimeError('Server stopped: '+log_path.read_text(errors='replace')[-6000:])
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/health',timeout=2) as response:
                print('Ready:',response.read().decode(),flush=True)
            break
        except OSError:time.sleep(1)
    else:raise TimeoutError('Check '+str(log_path))
    public_tunnel=ngrok.connect(PORT,'http')
    print('OPEN DEMO:',public_tunnel.public_url,flush=True)
    print('Models load lazily on first inference. Logs:',log_path,flush=True)
except Exception:
    api_process.terminate()
    log_file.close()
    raise
'''),cell('markdown','''
## Stop this demo
Do not run setup again while server is active. This cell closes only this
notebook's server/tunnel. Outputs/checkpoints/datasets are preserved.
'''),cell('code','''
if 'public_tunnel' in globals():ngrok.disconnect(public_tunnel.public_url)
if 'api_process' in globals() and api_process.poll() is None:
    api_process.terminate()
    try:api_process.wait(timeout=10)
    except subprocess.TimeoutExpired:api_process.kill()
if 'log_file' in globals():log_file.close()
print('Demo stopped. Runtime artifacts remain in',OUTPUT)
''')]
    for i,c in enumerate(cells):c['id']=f'demo-{i:03d}'
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.10.10'),demo_bundle_sha256=sha),cells=cells)
    p=Path(__file__).with_name('01_kaggle_lryolo_demo.ipynb');p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
    print(p,len(files),'embedded files',sha)
if __name__=='__main__':main()
