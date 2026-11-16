"""Shared Kaggle workflow. Imported by self-contained generated notebooks.

Only standard-library imports at module load. Nothing installs, trains or
downloads until a notebook explicitly calls the corresponding function.
"""
import copy
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

REPO_COMMIT = 'c48a53a74df9bd3fb6ae268fe9217fec9f114a91'
MMDET_COMMIT = 'b5ca9f17'
CLASSES = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
           'motorcycle', 'bicycle', 'traffic light', 'traffic sign')
COUNTS = dict(train=31217, val=5511, test_day=5258,
              test_night=3929, test_synthetic_dark=5258)
CONFIGS = dict(clean='01_clean.py', dark='02_dark.py', evaluate='03_evaluate.py')


def call(args, cwd=None, **kwargs):
    print('Running:', ' '.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=cwd, check=True, **kwargs)


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def clone_pinned(url, destination, commit):
    destination = Path(destination)
    if not destination.exists():
        call(['git', 'clone', url, destination])
        call(['git', 'checkout', commit], cwd=destination)
    actual = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=destination, text=True).strip()
    if not actual.startswith(commit):
        raise RuntimeError(f'{destination}: expected {commit}, got {actual}. '
                           'Use a separate checkout; existing files are preserved.')
    return destination


def bootstrap(repo, bundle, install_environment=False, mmcv_wheel=None, num_gpus=2):
    """Reuse the supplied notebook's versions; never delete a checkout.

    Set install_environment=True in a fresh pinned Kaggle session to execute
    the known setup recipe. A matching working stack is not reinstalled.
    """
    if sys.version_info[:3] != (3, 10, 10):
        raise RuntimeError(f'Use the pinned Python 3.10.10 Kaggle image: {sys.version}')
    torch_targets = {'torch': '2.5.1', 'torchvision': '0.20.1', 'torchaudio': '2.5.1'}
    mismatch = any((version(p) or '').split('+')[0] != v
                   for p, v in torch_targets.items())
    if mismatch:
        if not install_environment:
            raise RuntimeError('Torch stack differs. In a fresh session, enable '
                               'INSTALL_ENVIRONMENT to run the original pinned recipe.')
        if any(p in sys.modules for p in torch_targets):
            raise RuntimeError('Restart the kernel before changing imported Torch packages.')
        call([sys.executable, '-m', 'pip', 'install', '-q',
              '--index-url', 'https://pypi.org/simple',
              'typing_extensions>=4.8', 'flit_core>=3.11,<4'])
        call([sys.executable, '-m', 'pip', 'install', '-q',
              '--index-url', 'https://pypi.org/simple', '--extra-index-url',
              'https://download.pytorch.org/whl/cu121',
              *[f'{p}=={v}' for p, v in torch_targets.items()]])
    required = {'mmengine': '0.10.7', 'pycocotools': None,
                'terminaltables': None, 'lpips': None, 'einops': None, 'gdown': None}
    missing = [f'{p}=={v}' if v else p for p, v in required.items()
               if version(p) is None or (v and version(p) != v)]
    if missing:
        if not install_environment:
            raise RuntimeError(f'Missing/different packages: {missing}. '
                               'Enable INSTALL_ENVIRONMENT in a fresh session.')
        call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
    if version('mmcv') != '2.0.1':
        if not install_environment:
            raise RuntimeError('MMCV 2.0.1 required; attach the tested T4 wheel.')
        candidates = ([Path(mmcv_wheel)] if mmcv_wheel else
                      list(Path('/kaggle/input').rglob('mmcv*torch2.5.1*cu121*t4*.whl')))
        if len(candidates) != 1:
            raise RuntimeError(f'Set MMCV_WHEEL explicitly; candidates: {candidates}')
        wheel = Path('/kaggle/working/mmcv-2.0.1-cp310-cp310-linux_x86_64.whl')
        content = candidates[0].read_bytes()
        if wheel.exists() and wheel.read_bytes() != content:
            raise FileExistsError(wheel)
        wheel.write_bytes(content)
        call([sys.executable, '-m', 'pip', 'install', '--no-deps', wheel])
    if version('mmdet') != '3.3.0':
        if not install_environment:
            raise RuntimeError('The original notebook uses custom MMDetection 3.3.0.')
        source = clone_pinned('https://github.com/diasuki/mmdetection.git',
                              '/kaggle/working/mmdetection', MMDET_COMMIT)
        call([sys.executable, '-m', 'pip', 'install', '.', '--no-deps',
              '--no-build-isolation'], cwd=source)
    repo = clone_pinned('https://github.com/leeminsun1205/LR-YOLO.git',
                        repo, REPO_COMMIT)
    # New project files are bundled because they are not committed/published.
    for relative in sorted(Path(bundle).rglob('*.py')):
        rel = relative.relative_to(bundle)
        if rel.parts[0] not in ('configs', 'projects'):
            continue
        destination = repo / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != relative.read_bytes():
            raise FileExistsError(f'Preserving differing existing file: {destination}')
        if not destination.exists():
            destination.write_bytes(relative.read_bytes())
    if version('mmyolo') is None:
        if not install_environment:
            raise RuntimeError('MMYOLO is missing; enable INSTALL_ENVIRONMENT.')
        call([sys.executable, '-m', 'pip', 'install', '.', '--no-deps',
              '--no-build-isolation'], cwd=repo)
    sys.path.insert(0, str(repo))
    os.environ['PYTHONPATH'] = str(repo) + os.pathsep + os.environ.get('PYTHONPATH', '')
    import torch
    import torchvision
    import torchaudio
    import mmcv
    from mmcv.ops import nms
    import mmengine
    import mmdet
    from mmdet.utils import setup_cache_size_limit_of_dynamo
    import mmyolo
    assert torch.version.cuda == '12.1', torch.version.cuda
    assert torch.cuda.is_available(), 'GPU required by the original LPIPS constructor.'
    assert torch.cuda.device_count() >= num_gpus, (
        f'Requested {num_gpus} GPUs; found {torch.cuda.device_count()}. Select T4 x2.')
    assert Path(mmyolo.__file__).resolve().is_relative_to(repo.resolve()), mmyolo.__file__
    boxes = torch.tensor([[0., 0., 10., 10.]], device='cuda')
    nms(boxes, torch.ones(1, device='cuda'), 0.5)
    print('Runtime:', torch.__version__, torchvision.__version__, torchaudio.__version__,
          mmcv.__version__, mmengine.__version__, mmdet.__version__, mmyolo.__version__)
    print('MMYOLO source:', mmyolo.__file__)
    print('GPUs:', [torch.cuda.get_device_name(i) for i in range(num_gpus)])
    return repo


def find_data_root(explicit=None):
    if explicit:
        root = Path(explicit)
    else:
        candidates = set()
        for pattern in ('*/bdd100k_day_night_voc/classes.txt',
                        'datasets/*/*/bdd100k_day_night_voc/classes.txt'):
            candidates.update(p.parent for p in Path('/kaggle/input').glob(pattern))
        if len(candidates) != 1:
            raise RuntimeError(f'Set DATA_ROOT explicitly. Candidates: {sorted(candidates)}')
        root = candidates.pop()
    assert (root / 'classes.txt').is_file(), root
    return root


def audit_dataset(root, output):
    """Check all IDs/protocol; sample XML/images without scanning every pixel."""
    root = Path(root)
    assert tuple((root / 'classes.txt').read_text().splitlines()) == CLASSES
    ids = {}
    report = {'root': str(root), 'splits': {}, 'classes': CLASSES}
    for split, expected in COUNTS.items():
        manifest = root / split / 'ImageSets/Main' / f'{split}.txt'
        values = manifest.read_text().split()
        assert len(values) == len(set(values)) == expected, (split, len(values))
        ids[split] = set(values)
        for image_id in (values[0], values[len(values) // 2], values[-1]):
            image = root / split / 'JPEGImages' / f'{image_id}.jpg'
            xml = root / split / 'Annotations' / f'{image_id}.xml'
            from PIL import Image
            with Image.open(image) as sample:
                sample.load()
                size = sample.size
            tree = ET.parse(xml).getroot()
            assert size == (int(tree.findtext('size/width')),
                            int(tree.findtext('size/height')))
            assert all(o.findtext('name') in CLASSES for o in tree.findall('object'))
        report['splits'][split] = dict(images=len(values), manifest_sha256=sha256(manifest))
    independent = ('train', 'val', 'test_day', 'test_night')
    for index, a in enumerate(independent):
        for b in independent[index + 1:]:
            assert not ids[a] & ids[b], (a, b)
    assert ids['test_day'] == ids['test_synthetic_dark']
    import csv
    gamma_file = root / 'test_synthetic_dark/gamma_values.csv'
    with gamma_file.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len({r['image_id'] for r in rows}) == COUNTS['test_day']
    assert {r['image_id'] for r in rows} == ids['test_day']
    assert all(1.5 <= float(r['gamma']) <= 5 for r in rows)
    report['gamma_sha256'] = sha256(gamma_file)
    report['validation_protocol'] = 'online random gamma; no fixed validation view'
    report['audit_scope'] = 'all manifest IDs, three decoded image/XML pairs per split'
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return report


def load_torch_checkpoint(path):
    """Use the original notebook's temporary NumPy aliases only when needed."""
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except ModuleNotFoundError as exc:
        if not str(exc.name).startswith('numpy._core'):
            raise
        import numpy.core
        aliases = {'numpy._core': numpy.core,
                   'numpy._core.multiarray': numpy.core.multiarray}
        old = {key: sys.modules.get(key) for key in aliases}
        try:
            sys.modules.update(aliases)
            return torch.load(path, map_location='cpu', weights_only=False)
        finally:
            for key, value in old.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value


def prepare_transfer_checkpoint(source, destination, download_release=False):
    """Retain VOC feature/regression weights, discard both class branches."""
    import torch
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source is None:
        if not download_release:
            raise ValueError('Set VOC_CHECKPOINT to a trusted mounted checkpoint, '
                             'or explicitly set DOWNLOAD_RELEASE=True.')
        import zipfile
        import gdown
        bundle = destination.parent / 'lr_yolov10_s_voc_dark_bundle.zip'
        if not bundle.exists():
            gdown.download(id='1eIwemKii0QJmSaBByYKZINmmVX69Xtcs',
                           output=str(bundle), quiet=False)
        source = destination.parent / 'released_best_epoch_97.pth'
        if not source.exists():
            with zipfile.ZipFile(bundle) as archive:
                matches = [n for n in archive.namelist()
                           if n.endswith('best_pascal_voc_mAP_epoch_97.pth')]
                assert len(matches) == 1, matches
                with archive.open(matches[0]) as src, source.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
    source = Path(source)
    checkpoint = load_torch_checkpoint(source)
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    discarded = [k for k in state if k.startswith((
        'bbox_head.head_module.one2many_cls_preds.',
        'bbox_head.head_module.one2one_cls_preds.'))]
    if not discarded or not any(k.startswith('backbone.') for k in state):
        raise ValueError('Not the expected trusted MMYOLO LR-YOLOv10 checkpoint.')
    report = dict(source=str(source), source_sha256=sha256(source),
                  removed_keys=discarded, target_classes=CLASSES,
                  reason='VOC and BDD classes differ; learn new BDD classifiers in clean stage')
    sidecar = destination.with_suffix('.json')
    if destination.exists():
        if not sidecar.exists() or json.loads(sidecar.read_text()) != json.loads(json.dumps(report)):
            raise FileExistsError(f'Preserving existing checkpoint: {destination}')
        return destination
    torch.save(dict(state_dict={k: v for k, v in state.items() if k not in discarded},
                    meta={'bdd_transfer': report}), destination)
    write_json(sidecar, report)
    print('Removed classifier tensors/buffers:', len(discarded))
    return destination


def require_bdd_checkpoint(path):
    checkpoint = load_torch_checkpoint(path)
    meta = checkpoint.get('meta', {})
    classes = meta.get('dataset_meta', {}).get('classes', ())
    if tuple(classes) != CLASSES:
        raise ValueError('Checkpoint must have BDD dataset_meta classes in the '
                         'specified order. VOC transfer-only checkpoints are not valid here.')
    return Path(path)


def clean_initialization(mode, output_dir, source=None, download_release=False,
                         resume_checkpoint=None):
    """Random detector initialization or explicitly requested VOC transfer."""
    if mode not in ('scratch', 'voc_transfer'):
        raise ValueError('INITIALIZATION must be scratch or voc_transfer.')
    if resume_checkpoint is not None:
        return require_bdd_checkpoint(resume_checkpoint)
    if mode == 'scratch':
        print('Scratch: detector/reconstruction initialize randomly; no VOC checkpoint.')
        return None
    return prepare_transfer_checkpoint(
        source, Path(output_dir) / 'initialization/voc_features_without_classifiers.pth',
        download_release=download_release)


def subset_manifest(root, split, count, output, seed):
    source = Path(root) / split / 'ImageSets/Main' / f'{split}.txt'
    values = source.read_text().split()
    if count is None:
        return str(source)
    assert 0 < count <= len(values)
    chosen = set(random.Random(seed).sample(values, count))
    selected = [item for item in values if item in chosen]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(selected) + '\n'
    if output.exists() and output.read_text() != text:
        raise FileExistsError(output)
    output.write_text(text)
    return str(output)


def training_config(repo, root, stage, work_dir, *, init_checkpoint=None,
                    epochs=3, batch=2, workers=2, seed=42, lr=0.01,
                    train_limit=None, val_limit=None, short_schedule=False,
                    resume=False, num_gpus=2, deadline_unix=None):
    from mmengine.config import Config
    assert stage in ('clean', 'dark')
    assert epochs > 0 and batch > 0 and workers >= 0
    work_dir = Path(work_dir)
    if stage == 'dark' or resume:
        require_bdd_checkpoint(init_checkpoint)
    if resume:
        saved = load_torch_checkpoint(init_checkpoint)
        if 'optimizer' not in saved:
            raise ValueError('Resume needs an epoch_N.pth with optimizer state; '
                             'best checkpoints are for initialization/evaluation.')
        if saved.get('meta', {}).get('epoch', 0) >= epochs:
            raise ValueError('Total epochs must exceed the saved epoch when resuming.')
    cfg = Config.fromfile(str(Path(repo) / 'configs/yolov10_bdd100k_day_night' / CONFIGS[stage]))
    for split, loader in [('train', cfg.train_dataloader), ('val', cfg.val_dataloader)]:
        loader.dataset.data_root = str(root) + '/'
        loader.dataset.ann_file = subset_manifest(
            root, split, train_limit if split == 'train' else val_limit,
            work_dir / f'{split}_ids.txt', seed)
        loader.dataset.data_prefix.sub_data_root = split + '/'
        loader.num_workers = workers
        loader.persistent_workers = workers > 0
    cfg.train_dataloader.batch_size = batch
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.train_batch_size_per_gpu = batch
    cfg.train_num_workers = cfg.val_num_workers = workers
    cfg.max_epochs = epochs
    cfg.base_lr = lr
    cfg.optim_wrapper.optimizer.update(lr=lr, batch_size_per_gpu=batch)
    cfg.train_cfg.update(max_epochs=epochs, val_interval=1, dynamic_intervals=[])
    cfg.default_hooks.param_scheduler.max_epochs = epochs
    if short_schedule:
        cfg.default_hooks.param_scheduler.update(
            warmup_epochs=min(1., epochs / 3), warmup_mim_iter=0)
    cfg.randomness.seed = seed
    cfg.work_dir = str(work_dir)
    cfg.load_from = str(init_checkpoint) if init_checkpoint else None
    cfg.resume = resume
    cfg.stage_checkpoint_selection = 'last' if stage == 'clean' else 'best'
    cfg.num_gpus = num_gpus
    cfg.launcher = 'pytorch' if num_gpus > 1 else 'none'
    for hook in cfg.custom_hooks:
        if hook['type'] == 'BDDSessionBudgetHook':
            hook['deadline_unix'] = deadline_unix
    if resume:
        from projects.bdd100k.state import training_signature
        original = saved.get('meta', {}).get('bdd_run')
        if original != training_signature(cfg):
            raise ValueError('Resume signature differs: keep stage, total epochs, '
                             'batch, GPU count, LR/schedule, seed and subsets unchanged.')
    assert cfg.model.bbox_head.head_module.num_classes == len(CLASSES)
    assert cfg.model.data_preprocessor.type == (
        'YOLOv5DarkDetDataPreprocessor' if stage == 'dark' else 'YOLOv5DetDataPreprocessor')
    return cfg


def training_command(config_file, num_gpus):
    if num_gpus > 1:
        return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={num_gpus}', 'tools/train.py', str(config_file),
                '--launcher', 'pytorch']
    return [sys.executable, 'tools/train.py', str(config_file), '--launcher', 'none']


def copy_resume_best(source, work_dir):
    """Carry previous best score/file forward without touching read-only inputs."""
    checkpoint = load_torch_checkpoint(source)
    hub = checkpoint.get('message_hub', {}).get('runtime_info', {})
    previous = hub.get('best_ckpt')
    if previous:
        import shutil
        previous_file = Path(source).parent / Path(previous).name
        if not previous_file.is_file():
            raise FileNotFoundError('Attach the entire previous stage output folder, '
                                    f'including {previous_file.name}, when resuming.')
        destination = Path(work_dir) / previous_file.name
        if destination.exists() and sha256(destination) != sha256(previous_file):
            raise FileExistsError(destination)
        if not destination.exists():
            shutil.copy2(previous_file, destination)


def inspect_contract(cfg, output_dir):
    """Small GPU check of loader coordinates, pairing, and freeze policy."""
    import gc
    import numpy as np
    import torch
    from mmengine.runner import Runner, set_random_seed
    from mmyolo.registry import MODELS
    from mmyolo.utils import register_all_modules
    from projects.bdd100k.dataset import read_bdd_boxes
    from torchvision.utils import save_image
    register_all_modules()
    set_random_seed(cfg.randomness.seed, deterministic=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader_cfg = copy.deepcopy(cfg.train_dataloader)
    # Inspect at most ten training images even for a full-run configuration.
    selected = Path(loader_cfg.dataset.ann_file).read_text().split()[:10]
    manifest = output_dir / 'contract_train_ids.txt'
    manifest.write_text('\n'.join(selected) + '\n')
    loader_cfg.dataset.ann_file = str(manifest)
    loader = Runner.build_dataloader(loader_cfg, seed=cfg.randomness.seed)
    info = loader.dataset.get_data_info(0)
    image = Path(info['img_path'])
    boxes = read_bdd_boxes(image.parent.parent / 'Annotations' / f'{image.stem}.xml', CLASSES)
    assert all(i['bbox_label'] == label and np.allclose(i['bbox'], xyxy)
               for i, (label, xyxy) in zip(info['instances'], boxes))
    model = MODELS.build(cfg.model).cuda()
    missing = []
    if cfg.load_from:
        saved = load_torch_checkpoint(cfg.load_from)
        model.init_weights()
        incompatibility = model.load_state_dict(saved['state_dict'], strict=False)
        missing = list(incompatibility.missing_keys)
        transfer = 'bdd_transfer' in saved.get('meta', {})
        allowed = ('bbox_head.head_module.one2many_cls_preds.',
                   'bbox_head.head_module.one2one_cls_preds.') if transfer else ()
        if incompatibility.unexpected_keys or any(not k.startswith(allowed) for k in missing):
            raise ValueError(f'Unexpected checkpoint mismatch: {incompatibility}')
    data = model.data_preprocessor(next(iter(loader)), training=True)
    x = data['inputs']
    assert x.ndim == 4 and tuple(x.shape[-2:]) == (640, 640)
    assert bool(torch.isfinite(x).all()) and 0 <= x.min() <= x.max() <= 1
    dark = cfg.model.type == 'YOLOReconstructDehazeDetector'
    if dark:
        gts = data['gts']
        assert x.shape == gts.shape and bool((x <= gts + 1e-6).all())
        assert not torch.allclose(x, gts)
        assert all(not p.requires_grad for p in model.bbox_head.parameters())
        assert all(not p.requires_grad for p in model.neck.parameters())
        preview = torch.cat([gts[:2], x[:2]])
    else:
        assert all(p.requires_grad for p in model.bbox_head.parameters())
        preview = x[:2]
    save_image(preview, output_dir / 'training_inputs.png', nrow=2)
    report = dict(model=cfg.model.type, input_shape=list(x.shape),
                  clean_gts='gts' in data,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  bbox_head_trainable=any(p.requires_grad for p in model.bbox_head.parameters()),
                  coordinate_check='zero-based XML and loaded boxes agree',
                  initialization_missing_keys=missing,
                  validation='online random gamma' if dark else 'clean daytime')
    write_json(output_dir / 'contract.json', report)
    print(report)
    del model, data, x, preview, loader
    gc.collect()
    torch.cuda.empty_cache()
    return report


def provenance(repo, cfg, config_file):
    import platform
    import torch
    status = subprocess.check_output(['git', 'status', '--short'], cwd=repo, text=True)
    record = dict(python=sys.version, platform=platform.platform(), seed=cfg.randomness.seed,
                  repo_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                                      text=True).strip(),
                  repo_status=status,
                  gpus=[torch.cuda.get_device_name(i) for i in range(cfg.num_gpus)],
                  world_size=cfg.num_gpus,
                  global_batch=cfg.train_dataloader.batch_size * cfg.num_gpus,
                  cuda=torch.version.cuda, config_sha256=sha256(config_file),
                  versions={p: version(p) for p in ('torch', 'torchvision', 'torchaudio',
                            'mmcv', 'mmengine', 'mmdet', 'mmyolo', 'numpy', 'lpips')},
                  initialization=str(cfg.load_from), resume=cfg.resume)
    if cfg.load_from:
        record['initialization_sha256'] = sha256(cfg.load_from)
    for split, loader in [('train', cfg.train_dataloader), ('val', cfg.val_dataloader)]:
        record[f'{split}_manifest_sha256'] = sha256(loader.dataset.ann_file)
    record['project_sources'] = {
        str(p.relative_to(repo)): sha256(p)
        for folder in ('configs/yolov10_bdd100k_day_night', 'projects/bdd100k')
        for p in (Path(repo) / folder).glob('*.py')}
    write_json(Path(cfg.work_dir) / 'provenance.json', record)


def run_stage(repo, cfg):
    work_dir = Path(cfg.work_dir)
    if list(work_dir.glob('*.pth')) and not cfg.resume:
        raise FileExistsError('Run already has checkpoints. Use a new RUN_ID or explicit resume.')
    work_dir.mkdir(parents=True, exist_ok=True)
    config_file = work_dir / 'resolved_config.py'
    if config_file.exists():
        raise FileExistsError('Preserving previous resolved config. Use a new run directory.')
    for hook in cfg.custom_hooks:
        if hook['type'] == 'BDDSessionBudgetHook':
            deadline = hook.get('deadline_unix')
            if deadline is not None and time.time() + hook.get('reserve_seconds', 1800) >= deadline:
                print('Insufficient session budget to start another stage. Save outputs and continue later.')
                return None
    if cfg.resume:
        copy_resume_best(cfg.load_from, work_dir)
    cfg.dump(str(config_file))
    provenance(repo, cfg, config_file)
    call(training_command(config_file, cfg.num_gpus), cwd=repo)
    session = work_dir / 'session_state.json'
    if session.exists() and json.loads(session.read_text())['status'] == 'session_paused':
        print('Stage paused cleanly. Save this output folder and resume from its latest epoch checkpoint.')
        return None
    if cfg.get('stage_checkpoint_selection') == 'last':
        # Released stage-3 config explicitly loads clean epoch_100, not its best.
        final_checkpoint = work_dir / f'epoch_{cfg.train_cfg.max_epochs}.pth'
        if not final_checkpoint.is_file():
            raise FileNotFoundError(final_checkpoint)
        return require_bdd_checkpoint(final_checkpoint)
    best = list(work_dir.glob('best_pascal_voc_mAP_epoch_*.pth'))
    if len(best) != 1:
        raise RuntimeError(f'Expected one best checkpoint selected on internal val: {best}')
    return require_bdd_checkpoint(best[0])


def smoke_test(repo, root, output_dir, seed=42, num_gpus=2):
    """One epoch per original stage, ten train and ten internal-val images."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Use a fresh smoke directory: {output_dir}')
    clean = training_config(repo, root, 'clean', output_dir / 'clean',
                            epochs=1, workers=0, seed=seed,
                            train_limit=10, val_limit=10, short_schedule=True,
                            num_gpus=num_gpus)
    inspect_contract(clean, output_dir / 'clean/contract')
    clean_checkpoint = run_stage(repo, clean)
    dark = training_config(repo, root, 'dark', output_dir / 'dark',
                           init_checkpoint=clean_checkpoint, epochs=1,
                           workers=0, seed=seed, train_limit=10,
                           val_limit=10, short_schedule=True, num_gpus=num_gpus)
    inspect_contract(dark, output_dir / 'dark/contract')
    dark_checkpoint = run_stage(repo, dark)
    # A tiny backward pass proves the loss contract; smoke scores are not results.
    report = dict(status='passed', scope='10-image clean + dark training/validation',
                  clean_checkpoint=str(clean_checkpoint), dark_checkpoint=str(dark_checkpoint),
                  scientific_result=False)
    write_json(output_dir / 'smoke_passed.json', report)
    return report


def profile_batches(repo, root, output_dir, candidates=(2, 4, 8, 16),
                    num_gpus=2, seed=42, memory_fraction=0.90):
    """Six scratch steps per stage/batch in fresh DDP processes; no checkpoint.

    Reports actual full-precision runtime/VRAM. Only CUDA OOM is treated as an
    infeasible batch; other errors stop the profiling workflow.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    report = dict(world_size=num_gpus, candidates=list(candidates),
                  target_global_batch=16, precision='fp32', runs=[],
                  initialization='scratch; zero-LR backward/optimizer resource probe')
    for batch in candidates:
        for stage in ('clean', 'dark'):
            folder = output_dir / f'{stage}_batch_{batch}'
            # Build dataset/schedule via clean config, then switch the model only
            # for a scratch resource probe of the original dark-stage graph.
            cfg = training_config(
                repo, root, 'clean', folder, epochs=1, batch=batch, workers=0,
                train_limit=batch * num_gpus * 6, val_limit=10,
                seed=seed, short_schedule=True, num_gpus=num_gpus)
            if stage == 'dark':
                cfg.model.type = 'YOLOReconstructDehazeDetector'
                cfg.model.data_preprocessor.type = 'YOLOv5DarkDetDataPreprocessor'
            # Allocate momentum/gradient state but do not optimize scratch weights.
            cfg.optim_wrapper.optimizer.lr = cfg.base_lr = 0.0
            cfg.default_hooks.param_scheduler.warmup_bias_lr = 0.0
            # The pinned tools/train.py metainfo helper cannot iterate None
            # dataloaders. Missing keys still disable Runner loops correctly.
            for key in ('val_dataloader', 'val_cfg', 'val_evaluator',
                        'test_dataloader', 'test_cfg', 'test_evaluator'):
                cfg.pop(key, None)
            cfg.default_hooks.checkpoint = None
            cfg.custom_hooks = [h for h in cfg.custom_hooks
                                if h['type'] in ('EMAHook', 'BDDRunDiagnosticsHook')]
            for hook in cfg.custom_hooks:
                if hook['type'] == 'BDDRunDiagnosticsHook':
                    hook['record_step_times'] = True
            config_file = folder / 'profile_config.py'
            cfg.dump(str(config_file))
            log = folder / 'profile.log'
            row = dict(stage=stage, batch_per_gpu=batch, global_batch=batch * num_gpus)
            try:
                with log.open('w') as handle:
                    call(training_command(config_file, num_gpus), cwd=repo,
                         stdout=handle, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                original_log = log.read_text(errors='replace')
                text = original_log.lower()
                if not any(message in text for message in (
                        'cuda out of memory', 'cuda error: out of memory')):
                    tail = '\n'.join(original_log.splitlines()[-100:])
                    raise RuntimeError(
                        f'Profiling failed for a non-OOM reason: {log}\n'
                        f'--- Last 100 log lines ---\n{tail}')
                row.update(status='oom', log=str(log))
            else:
                ranks = [json.loads((folder / f'compute_rank_{rank}.json').read_text())
                         for rank in range(num_gpus)]
                row.update(status='ok', ranks=ranks,
                           reserved_fraction=max(r['peak_reserved_bytes'] / r['device_total_bytes']
                                                 for r in ranks),
                           global_images_per_second=min(r['global_images_per_second'] for r in ranks))
            report['runs'].append(row)
            write_json(output_dir / 'batch_profile.json', report)
            print(stage, batch, row['status'])
    safe = [batch for batch in candidates
            if all(any(r['stage'] == stage and r['batch_per_gpu'] == batch
                       and r['status'] == 'ok' and r['reserved_fraction'] <= memory_fraction
                       for r in report['runs']) for stage in ('clean', 'dark'))]
    preferred = [batch for batch in safe if batch * num_gpus <= 16]
    report['recommended_batch_per_gpu'] = max(preferred) if preferred else None
    report['memory_headroom_fraction'] = 1 - memory_fraction
    write_json(output_dir / 'batch_profile.json', report)
    print('Profile:', output_dir / 'batch_profile.json')
    if report['recommended_batch_per_gpu'] is None:
        raise RuntimeError('No batch passed both stages with headroom; inspect the profile.')
    return report


def evaluate(repo, root, checkpoint, output_dir, seed=42, workers=2):
    """Evaluate a preselected BDD checkpoint on all three domains unchanged."""
    from mmengine.config import Config
    require_bdd_checkpoint(checkpoint)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Use a new evaluation output directory: {output_dir}')
    output_dir.mkdir(parents=True)
    record = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                  selection='chosen using internal val before inspecting test metrics',
                  metric='VOC AP50, 11points, all ten BDD classes', seed=seed,
                  repo_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                                      text=True).strip(),
                  workflow_sha256=sha256(__file__), results={})
    for split in ('test_day', 'test_synthetic_dark', 'test_night'):
        cfg = Config.fromfile(str(Path(repo) / 'configs/yolov10_bdd100k_day_night/03_evaluate.py'))
        # tools/test.py uses the same metainfo helper as tools/train.py.
        cfg.pop('train_dataloader', None)
        cfg.pop('val_dataloader', None)
        cfg.test_dataloader.dataset.update(
            data_root=str(root) + '/', ann_file=f'{split}/ImageSets/Main/{split}.txt',
            data_prefix=dict(sub_data_root=split + '/'))
        cfg.test_dataloader.update(num_workers=workers, persistent_workers=workers > 0)
        cfg.randomness.seed = seed
        cfg.work_dir = str(output_dir / split)
        assert cfg.model.data_preprocessor.type == 'YOLOv5DetDataPreprocessor'
        config_file = output_dir / f'{split}.py'
        cfg.dump(str(config_file))
        call([sys.executable, 'tools/test.py', config_file, checkpoint], cwd=repo)
        metrics = sorted(Path(cfg.work_dir).glob('*/**/metrics.json'))
        record['results'][split] = dict(config_sha256=sha256(config_file),
                                       work_dir=cfg.work_dir,
                                       metric_files=[str(p) for p in metrics])
        # MMEngine also writes timestamp.json; retain all original per-class tables/logs.
        logs = list(Path(cfg.work_dir).rglob('*.json'))
        scalar_rows = []
        for path in logs:
            for line in path.read_text().splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and 'pascal_voc/mAP' in value:
                    scalar_rows.append(value)
        if not scalar_rows:
            raise RuntimeError(f'No AP50 result found in logs: {cfg.work_dir}')
        record['results'][split]['metrics'] = scalar_rows[-1]
        write_json(output_dir / 'summary.json', record)
    print(json.dumps(record, indent=2))
    return record
