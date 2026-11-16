"""One-notebook VOC_Dark adapter experiment registry and orchestration.

Standard library at import time. GPU/model work occurs only on explicit calls.
Dataset downloads occur only via explicitly called preparation with permission
in notebook config. No resplitting, original source patches or full LR training.
"""
import copy
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import random
import shutil
import stat
import subprocess
import sys
import tarfile
import uuid
import xml.etree.ElementTree as ET
import zipfile

import bdd100k_workflow as shared

VOC_CLASSES = ('person', 'car', 'bus', 'bicycle', 'motorbike',
               'boat', 'bottle', 'chair', 'dog', 'cat')
EXDARK_CLASSES = ('People', 'Car', 'Bus', 'Bicycle', 'Motorbike',
                  'Boat', 'Bottle', 'Chair', 'Dog', 'Cat')
# Released VOC XMLs may retain objects from the original 20-class vocabulary.
# The VOC loader filters these by METAINFO, without changing the source files.
VOC_EXTRA_CLASSES = frozenset(('aeroplane', 'bird', 'cow', 'diningtable', 'horse',
                               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'))
ORIGINAL_CONFIG = ('configs/yolov10_voc_dark/'
                   '3_finetune_yolov10_s_reconstruct_voc_dark_ep100_dark_train_no_aug_dark_test.py')
CASES = {
    0: dict(name='released_full_decoder', adapter='none', decoder_free=False,
            train=False, purpose='Rerun released detection weights; full decoder executes'),
    1: dict(name='released_decoder_free', adapter='none', decoder_free=True,
            train=False, purpose='Equivalence/latency control; no adapter'),
    2: dict(name='scale_shift', adapter='scale_shift', decoder_free=True,
            train=True, purpose='Lightweight per-channel affine adaptation control'),
    3: dict(name='residual_adapter', adapter='residual', decoder_free=True,
            train=True, purpose='Primary bottleneck adapter candidate'),
    4: dict(name='gated_adapter', adapter='gated', decoder_free=True,
            train=True, purpose='Feature-statistics gate ablation, compared against case 3'),
}
CHECKPOINTS = {
    0: 'Official VOC_Dark LR-YOLOv10-S release, best_pascal_voc_mAP_epoch_97.pth',
    1: 'Explicit trusted full VOC LR-YOLOv10-S checkpoint (same architecture/classes)',
    2: 'Checkpoint created by notebook 05; metadata must match selected CASE_ID',
}
DATA_RELEASES = {
    'voc': dict(file_id='1jBSPYTNiP5-5fUtVjdkP4KZ2i6ZQpnbj', count=3760,
                archive_names=('VOC_Dark_Val.zip',)),
    'train': dict(file_id='1NEQTtCUNK5QlA1_Fh1s7U6h_CfcCP-gO', count=12334,
                  archive_names=('VOC_Dark.zip', 'VOC_Dark_Train.zip')),
    'exdark': dict(file_id='1MmgT0-mc_Q_mhnqBF5fpC3QaNntvgL69', count=2563,
                   archive_names=('ExDark.zip',)),
}
PRE_LOSS_BRIDGE_SHA256 = 'dd35b91f765d6ace0d0aff88dba767f74a53d124cd9a43de22f3e2b18fb7ee0b'


def migrate_loss_bridge(repo, bundle):
    """Upgrade only the exact known project version; preserve unknown edits.

    Needed before bootstrap's normal differing-file guard in an existing Kaggle
    session. Original mmyolo sources and checkpoint files are never touched.
    """
    relative = Path('projects/lowlight_adapter/models.py')
    destination, source = Path(repo) / relative, Path(bundle) / relative
    if not destination.is_file():
        return False  # Fresh checkout will receive the bundled project files.
    old, new = destination.read_bytes(), source.read_bytes()
    changed = old != new
    if changed:
        if hashlib.sha256(old).hexdigest() != PRE_LOSS_BRIDGE_SHA256:
            raise FileExistsError(f'Preserving unknown project edits: {destination}. '
                                  'Use a separate REPO or review the difference.')
        compile(new.decode(), str(destination), 'exec')
        backup = destination.with_name(destination.name + '.before_loss_bridge_' + PRE_LOSS_BRIDGE_SHA256[:12])
        if backup.exists() and backup.read_bytes() != old:
            raise FileExistsError(backup)
        if not backup.exists():
            backup.write_bytes(old)
        destination.write_bytes(new)
        print('Updated known adapter loss bridge; backup:', backup)
    # Existing notebook kernel can still hold the old registered class object.
    cached = sys.modules.get('projects.lowlight_adapter.models')
    if cached is not None:
        tree = ast.parse(new.decode())
        detector = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                        and n.name == 'FrozenNeckAdapterDetector')
        method = next(n for n in detector.body if isinstance(n, ast.FunctionDef) and n.name == 'loss')
        namespace = {'__name__': cached.__name__}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(destination), 'exec'), namespace)
        cached.FrozenNeckAdapterDetector.loss = namespace['loss']
    return changed


def new_run_id(case_id, action, mode):
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    return f'case{case_id}_{action}_{mode}_{stamp}_{uuid.uuid4().hex[:6]}'


def required_domains(case_id, action, final_evaluation=False):
    if case_id not in CASES or action not in ('audit', 'smoke', 'train', 'evaluate'):
        raise ValueError('Unknown case/action')
    if action == 'train' and not CASES[case_id]['train']:
        raise ValueError('Cases 0/1 are not training cases')
    if final_evaluation and action != 'evaluate':
        raise ValueError('FINAL_EVALUATION requires ACTION=evaluate')
    domains = ['voc']
    if CASES[case_id]['train'] and action in ('audit', 'smoke', 'train'):
        domains.append('train')
    if final_evaluation:
        domains.append('exdark')
    return domains


def _manifest_spec(manifest, domain, release_specs=None):
    """Infer a VOC layout from metadata; do not inspect all images/XML contents."""
    manifest = Path(manifest)
    if not manifest.is_file():
        return None
    ids = manifest.read_text().split()
    releases = release_specs or DATA_RELEASES
    if len(ids) != releases[domain]['count'] or len(ids) != len(set(ids)):
        return None
    image_root = manifest.parent.parent.parent
    if not (image_root / 'JPEGImages').is_dir() or not (image_root / 'Annotations').is_dir():
        return None
    return dict(root=str(image_root.resolve()), manifest=str(manifest.resolve()), prefix='')


def discover_dataset(base, domain, max_depth=12, require_domain_hint=False, release_specs=None):
    """Walk directory metadata, pruning image/XML trees and unrelated caches."""
    base = Path(base)
    if not base.is_dir():
        return None
    found = {}
    excluded = {'JPEGImages', 'Annotations', 'images', '__MACOSX', '.git',
                '__pycache__', '.virtual_documents', 'node_modules'}
    for folder, directories, filenames in os.walk(base, followlinks=False):
        path = Path(folder)
        depth = len(path.relative_to(base).parts)
        directories[:] = sorted(d for d in directories if d not in excluded
                                 and not d.startswith('.') and '.extracting_' not in d
                                 and depth < max_depth)
        if path.name == 'Main' and path.parent.name == 'ImageSets':
            if require_domain_hint:
                # A clean VOC train bundle can also contain 3,760 test IDs.
                # Count alone must not misidentify it as fixed synthetic-dark val.
                parts = [p.lower().replace('-', '_') for p in path.parts]
                if release_specs and release_specs[domain].get('hints'):
                    hinted = any(any(h in p for h in release_specs[domain]['hints']) for p in parts)
                elif domain == 'voc':
                    hinted = any(p in ('voc_dark_test', 'voc_dark_val', 'voc_dark_validation')
                                 or p.startswith('voc_dark_val_') for p in parts)
                elif domain == 'train':
                    hinted = any(p in ('voc_dark', 'voc_dark_train')
                                 or p.startswith('voc_dark_train_') for p in parts)
                else:
                    hinted = any('exdark' in p or 'ex_dark' in p for p in parts)
                if not hinted:
                    continue
            for name in sorted(filenames):
                if name.endswith('.txt'):
                    spec = _manifest_spec(path / name, domain, release_specs)
                    if spec:
                        found[spec['manifest']] = spec
    if len(found) > 1:
        raise RuntimeError(f'Multiple {domain} manifests match; set root/manifest explicitly: '
                           f'{list(found)}')
    return next(iter(found.values()), None)


def _archive_inventory(path):
    """Validate archive type/paths before extracting any bytes."""
    if zipfile.is_zipfile(path):
        kind = 'zip'
        with zipfile.ZipFile(path) as archive:
            items = [(m.filename, m.file_size, m.is_dir(),
                      stat.S_ISLNK(m.external_attr >> 16)) for m in archive.infolist()]
    elif tarfile.is_tarfile(path):
        kind = 'tar'
        with tarfile.open(path) as archive:
            items = [(m.name, m.size, m.isdir(), not (m.isfile() or m.isdir()))
                     for m in archive.getmembers()]
    else:
        raise ValueError(f'Expected ZIP/TAR dataset archive, not HTML/RAR/other content: {path}')
    total, paths = 0, set()
    for name, size, directory, unsafe_type in items:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or unsafe_type:
            raise ValueError(f'Unsafe archive entry: {name}')
        if relative.parts and relative.parts[0] == '__MACOSX':
            continue
        if not directory:
            if str(relative) in paths:
                raise ValueError(f'Duplicate archive file: {name}')
            paths.add(str(relative))
            total += size
    return kind, total


def extract_dataset(archive_path, cache, domain, release_specs=None):
    """Extract to new staging, validate manifest, then promote. Never overwrite data."""
    archive_path, cache = Path(archive_path), Path(cache)
    kind, required_bytes = _archive_inventory(archive_path)
    digest = shared.sha256(archive_path)
    parent = cache / 'extracted' / domain
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / digest[:16]
    if destination.exists():
        spec = discover_dataset(destination, domain, release_specs=release_specs)
        if not spec:
            raise RuntimeError(f'Existing cache is incomplete; preserve it and use a new cache: {destination}')
        return spec
    if shutil.disk_usage(parent).free < required_bytes + 512 * 1024**2:
        raise RuntimeError('Insufficient free disk for extraction plus 512 MiB reserve. '
                           'Mount extracted data or choose another cache; nothing is deleted.')
    staging = parent / f'{digest[:16]}.extracting_{uuid.uuid4().hex[:8]}'
    staging.mkdir()
    if kind == 'zip':
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                relative = PurePosixPath(member.filename)
                if relative.parts and relative.parts[0] == '__MACOSX':
                    continue
                target = staging.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if relative.parts and relative.parts[0] == '__MACOSX':
                    continue
                target = staging.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
    spec = discover_dataset(staging, domain, release_specs=release_specs)
    if spec is None:
        raise RuntimeError(f'{domain} archive has no matching VOC layout/manifest '
                           f'({(release_specs or DATA_RELEASES)[domain]["count"]} IDs). Inspect {staging}; '
                           'no subset/conversion was generated.')
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)
    spec = discover_dataset(destination, domain, release_specs=release_specs)
    shared.write_json(destination / 'preparation.json', dict(
        archive=str(archive_path), archive_sha256=digest, extracted_bytes=required_bytes,
        expected_release_drive_id=(release_specs or DATA_RELEASES)[domain]['file_id'],
        archive_format=kind, domain=domain, dataset=spec))
    return spec


def prepare_datasets(overrides, cache, case_id, action, final_evaluation=False,
                     auto_download=True, archives=None, search_roots=None, domains=None,
                     release_specs=None):
    """Explicit notebook-time data preparation, limited to the selected workload.

    Priority: configured root, saved resolved path, known previous working paths,
    mounted extracted datasets, local/mounted ZIP, then official Drive release.
    Unneeded domains are untouched. No dependency installation or model work.
    """
    needed = required_domains(case_id, action, final_evaluation)
    releases = release_specs or DATA_RELEASES
    if domains is not None:
        if not domains or not set(domains).issubset(needed):
            raise ValueError('Requested domains must be a nonempty subset of the action data plan')
        needed = list(domains)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archives = archives or {}
    search_roots = [Path(p) for p in (search_roots if search_roots is not None else ['/kaggle/input'])]
    result = copy.deepcopy(overrides)
    legacy = {'voc': ['/kaggle/working/VOC_Dark_Val',
                       '/kaggle/working/LR-YOLO/data/VOC_Dark_Val',
                       '/kaggle/working/LR-YOLO/data/voc_dark_test'],
              'train': ['/kaggle/working/VOC_Dark', '/kaggle/working/LR-YOLO/data/voc_dark'],
              'exdark': ['/kaggle/working/ExDark', '/kaggle/working/LR-YOLO/data/ExDark']}
    if release_specs:
        legacy = {domain: [] for domain in releases}
    for domain in needed:
        settings = result.get(domain, {})
        spec, reused = None, None
        if settings.get('root') is not None:
            root = Path(settings['root'])
            if not root.is_dir():
                raise FileNotFoundError(f'Configured {domain} root does not exist: {root}')
            manifest = settings.get('manifest')
            if manifest is not None and settings.get('prefix') is not None:
                path = Path(manifest)
                if not path.is_absolute():
                    path = root / path
                if not _manifest_spec(path, domain, release_specs):
                    raise ValueError(f'Configured {domain} manifest/layout/count is invalid: {path}')
                if not all((root / settings['prefix'] / name).is_dir()
                           for name in ('JPEGImages', 'Annotations')):
                    raise ValueError(f'Configured {domain} prefix does not contain images/XML: '
                                     f'{root / settings["prefix"]}')
                spec = dict(root=str(root), manifest=str(path), prefix=settings['prefix'])
            else:
                spec = discover_dataset(root, domain, release_specs=release_specs)
            if spec is None:
                raise ValueError(f'No matching {domain} manifest in configured root: {root}')
            reused = 'configured_root'
        index = cache / f'{domain}_resolved.json'
        if spec is None and index.is_file():
            saved = json.loads(index.read_text())['dataset']
            if _manifest_spec(saved['manifest'], domain, release_specs):
                spec, reused = saved, 'saved_cache'
        if spec is None:
            locations = [(cache / 'extracted' / domain, False)]
            locations += [(Path(p), False) for p in legacy[domain]]
            locations += [(p, True) for p in search_roots]
            for root, require_hint in locations:
                spec = discover_dataset(root, domain, require_domain_hint=require_hint,
                                        release_specs=release_specs)
                if spec:
                    reused = str(root)
                    break
        if spec is None:
            source_archive = Path(archives[domain]) if archives.get(domain) else None
            if source_archive is None:
                candidates = []
                for root in [cache / 'archives', Path('/kaggle/working'), *search_roots]:
                    if root.is_dir():
                        for folder, dirs, files in os.walk(root):
                            depth = len(Path(folder).relative_to(root).parts)
                            dirs[:] = [d for d in dirs if depth < 5 and d not in (
                                'JPEGImages', 'Annotations', 'images', '.git', 'extracted')
                                and not d.startswith('.')]
                            for name in files:
                                if name.lower() in {n.lower() for n in releases[domain]['archive_names']}:
                                    candidates.append(Path(folder) / name)
                candidates = list(dict.fromkeys(candidates))
                if len(candidates) > 1:
                    raise RuntimeError(f'Multiple {domain} archives; set DATASET_ARCHIVES: {candidates}')
                source_archive = candidates[0] if candidates else None
            if source_archive is None:
                if not auto_download:
                    raise FileNotFoundError(f'{domain} not found; mount it or enable AUTO_DOWNLOAD_DATASETS')
                import gdown
                download_dir = cache / 'archives'
                download_dir.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(download_dir).free < 512 * 1024**2:
                    raise RuntimeError('Not enough free disk to start dataset download')
                source_archive = download_dir / releases[domain]['archive_names'][0]
                pending = download_dir / f'{source_archive.name}.downloading_{uuid.uuid4().hex[:8]}'
                print('Downloading official dataset:', domain, 'Drive ID:',
                      releases[domain]['file_id'], flush=True)
                gdown.download(id=releases[domain]['file_id'], output=str(pending), quiet=False)
                if not pending.is_file():
                    raise RuntimeError(f'Dataset download did not produce a file: {pending}')
                _archive_inventory(pending)  # Reject incomplete ZIP/HTML before cache promotion.
                if source_archive.exists():
                    raise FileExistsError(source_archive)
                pending.rename(source_archive)
            spec = extract_dataset(source_archive, cache, domain, release_specs)
            reused = str(source_archive)
        result[domain] = spec
        shared.write_json(index, dict(domain=domain, resolved_from=reused, dataset=spec,
                                     manifest_sha256=shared.sha256(spec['manifest'])))
        print(f'Dataset ready [{domain}]: {spec}', flush=True)
    return result


def selection(checkpoint_id, case_id, action):
    if checkpoint_id not in CHECKPOINTS or case_id not in CASES:
        raise ValueError('Unknown checkpoint/case ID; inspect the printed registries')
    if action not in ('audit', 'smoke', 'train', 'evaluate'):
        raise ValueError(action)
    if action == 'train' and not CASES[case_id]['train']:
        raise ValueError('Cases 0/1 are evaluation controls; choose smoke/evaluate')
    if action == 'evaluate' and CASES[case_id]['train'] and checkpoint_id != 2:
        raise ValueError('Evaluate trained adapters with CHECKPOINT_ID=2; use smoke for identity adapters')
    return dict(CASES[case_id], case_id=case_id, checkpoint_id=checkpoint_id,
                checkpoint_description=CHECKPOINTS[checkpoint_id], action=action)


def registry_rows():
    return [dict(case_id=k, **v) for k, v in CASES.items()]


def resolve_checkpoint(checkpoint_id, source, output, case_id, download_release=False):
    if checkpoint_id not in CHECKPOINTS:
        raise ValueError(checkpoint_id)
    if checkpoint_id == 0 and source is None:
        import zipfile
        if not download_release:
            raise ValueError('Mount SOURCE_CHECKPOINT or explicitly enable DOWNLOAD_RELEASE')
        import gdown
        folder = Path(output) / 'release'
        folder.mkdir(parents=True, exist_ok=True)
        bundle = folder / 'lr_yolov10_s_voc_dark_bundle.zip'
        if not bundle.exists():
            gdown.download(id='1eIwemKii0QJmSaBByYKZINmmVX69Xtcs',
                           output=str(bundle), quiet=False)
        source = folder / 'best_pascal_voc_mAP_epoch_97.pth'
        if not source.exists():
            with zipfile.ZipFile(bundle) as archive:
                members = [n for n in archive.namelist()
                           if n.endswith('best_pascal_voc_mAP_epoch_97.pth')]
                if len(members) != 1:
                    raise RuntimeError(f'Ambiguous release members: {members}')
                import shutil
                with archive.open(members[0]) as src, source.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
    if source is None or not Path(source).is_file():
        raise FileNotFoundError('Set SOURCE_CHECKPOINT to a trusted mounted .pth')
    source = Path(source)
    checkpoint = shared.load_torch_checkpoint(source)
    meta = checkpoint.get('meta', {})
    declared = meta.get('dataset_meta', {}).get('classes')
    if declared and tuple(declared) != VOC_CLASSES:
        raise ValueError('Checkpoint class semantics differ from VOC; BDD is not supported here')
    adaptation = meta.get('voc_adapter')
    if checkpoint_id == 2:
        if not adaptation or adaptation['case_id'] != case_id:
            raise ValueError('Adapted checkpoint metadata does not match CASE_ID')
    elif adaptation or any(k.removeprefix('module.').startswith('adapters.')
                           for k in checkpoint.get('state_dict', {})):
        raise ValueError('Use CHECKPOINT_ID=2 for an adapted checkpoint')
    state = checkpoint.get('state_dict')
    if not state or not any('one2one_cls_preds' in k for k in state):
        raise ValueError('Expected complete MMYOLO YOLOv10 VOC state_dict, not filtered BDD transfer')
    return source


def dataset_spec(root, manifest, prefix, domain, class_names=None):
    if domain not in ('train', 'voc', 'exdark'):
        raise ValueError(domain)
    if root is None:
        raise ValueError(f'Set the explicit mounted dataset root for {domain}')
    if manifest is None or prefix is None:
        raise ValueError(f'Set the actual manifest and relative image/XML prefix for {domain}')
    root = Path(root).resolve()
    if Path(prefix).is_absolute() or '..' in Path(prefix).parts:
        raise ValueError('Dataset prefix must be a relative directory inside root')
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    ids = manifest_path.read_text().split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'Empty/duplicate manifest: {manifest_path}')
    if any(Path(i).is_absolute() or '..' in Path(i).parts or '/' in i or '\\' in i for i in ids):
        raise ValueError('Expected plain VOC image IDs without directory traversal')
    classes = tuple(class_names) if class_names is not None else (EXDARK_CLASSES if domain == 'exdark' else VOC_CLASSES)
    # Read three XMLs and decode three images; count IDs, never enumerate all XMLs.
    from PIL import Image
    samples = []
    for image_id in dict.fromkeys((ids[0], ids[len(ids) // 2], ids[-1])):
        folder = root / prefix
        xml = folder / 'Annotations' / f'{image_id}.xml'
        tree = ET.parse(xml).getroot()
        filename = tree.findtext('filename')
        image = folder / 'JPEGImages' / (Path(filename).name if filename else f'{image_id}.jpg')
        if not image.is_file():
            candidates = list((folder / 'JPEGImages').glob(image_id + '.*'))
            if len(candidates) != 1:
                raise FileNotFoundError(f'Cannot resolve one image for {xml}: {candidates}')
            image = candidates[0]
        objects = tree.findall('object')
        unknown = {o.findtext('name') for o in objects} - set(classes)
        if domain in ('train', 'voc') or class_names is not None:
            unknown -= set(VOC_CLASSES) | VOC_EXTRA_CLASSES
        if unknown:
            raise ValueError(f'Wrong domain/class spelling in {xml}: {unknown}')
        target_objects = [o for o in objects if o.findtext('name') in classes]
        excluded = {}
        for obj in objects:
            name = obj.findtext('name')
            if name not in classes:
                excluded[name] = excluded.get(name, 0) + 1
        with Image.open(image) as opened:
            opened.load()
            size = opened.size
        samples.append(dict(id=image_id, image_size=size,
                            total_xml_objects=len(objects), objects=len(target_objects),
                            excluded_non_target_objects=excluded, difficult=sum(
                                int(o.findtext('difficult', '0')) != 0 for o in target_objects)))
    return dict(root=str(root), manifest=str(manifest_path), prefix=prefix,
                domain=domain, count=len(ids), manifest_sha256=shared.sha256(manifest_path),
                classes=classes, samples=samples)


def assert_no_train_val_overlap(train, val):
    train_ids = set(Path(train['manifest']).read_text().split())
    val_ids = set(Path(val['manifest']).read_text().split())
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f'Train/validation IDs overlap: {sorted(overlap)[:5]}')


def subset_spec(spec, limit, destination, seed):
    result = copy.deepcopy(spec)
    ids = Path(spec['manifest']).read_text().split()
    if limit is not None and limit < len(ids):
        if limit < 1:
            raise ValueError('Subset limit must be positive')
        ids = sorted(random.Random(seed).sample(ids, limit))
        destination = Path(destination)
        content = '\n'.join(ids) + '\n'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_text() != content:
            raise FileExistsError(destination)
        destination.write_text(content)
        result.update(manifest=str(destination), count=len(ids),
                      manifest_sha256=shared.sha256(destination))
    return result


def build_config(repo, case_id, work_dir, val, train=None, epochs=3,
                 batch=4, workers=2, seed=42, lr=0.001, num_gpus=2,
                 deadline_unix=None, reduction=16):
    from mmengine.config import Config
    if case_id not in CASES or epochs < 1 or batch < 1 or workers < 0:
        raise ValueError('Invalid experiment settings')
    case = CASES[case_id]
    cfg = Config.fromfile(str(Path(repo) / ORIGINAL_CONFIG))
    cfg.custom_imports = dict(imports=['projects.lowlight_adapter',
                                      'projects.bdd100k.hooks'], allow_failed_imports=False)
    cfg.model.type = 'FrozenNeckAdapterDetector'
    cfg.model.pop('loss_l1_weight', None)
    cfg.model.pop('loss_lpips_weight', None)
    cfg.model.update(adapter_kind=case['adapter'], reduction=reduction,
                     decoder_free=case['decoder_free'], use_syncbn=False)
    cfg.model.data_preprocessor.type = 'VOCAdapterDarkPreprocessor'
    cfg.model.data_preprocessor.gamma_seed = seed
    cfg.work_dir = str(work_dir)
    cfg.load_from = None
    cfg.resume = False
    cfg.randomness = dict(seed=seed, deterministic=False)
    cfg.launcher = 'pytorch' if num_gpus > 1 else 'none'
    cfg.find_unused_parameters = False
    cfg.max_epochs = epochs
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=1)
    cfg.val_cfg = dict(type='ValLoop')
    cfg.test_cfg = dict(type='TestLoop')
    cfg.val_evaluator = dict(type='mmdet.VOCMetric', metric='mAP',
                             eval_mode='11points', iou_thrs=0.5, prefix='voc_standard')
    cfg.test_evaluator = [copy.deepcopy(cfg.val_evaluator), dict(
        type='VOCIncludingDifficultMetric', metric='mAP', eval_mode='11points',
        iou_thrs=0.5, prefix='voc_all_objects')]

    def loader(spec, template, training):
        result = copy.deepcopy(template)
        result.update(batch_size=batch if training else 1,
                      num_workers=workers, persistent_workers=workers > 0)
        result.dataset.update(
            type='YOLOv5ExDarkDataset' if spec['domain'] == 'exdark' else 'YOLOv5VOCDarkDataset',
            data_root=spec['root'] + '/', ann_file=spec['manifest'],
            data_prefix=dict(sub_data_root=spec['prefix']),
            test_mode=not training)
        result.dataset.pop('batch_shapes_cfg', None)
        return result

    cfg.val_dataloader = loader(val, cfg.val_dataloader, False)
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    if train is not None:
        if train['domain'] != 'train' or val['domain'] != 'voc':
            raise ValueError('Adapter training uses VOC train and VOC internal validation only')
        cfg.train_dataloader = loader(train, cfg.train_dataloader, True)
    else:
        # Pinned is_metainfo_lower cannot handle explicit None loader values.
        cfg.pop('train_dataloader', None)
        cfg.train_cfg = None
    cfg.optim_wrapper = dict(type='OptimWrapper', clip_grad=dict(max_norm=10.),
                             constructor='AdapterOptimizerConstructor',
                             optimizer=dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0))
    cfg.param_scheduler = [dict(type='LinearLR', by_epoch=True, begin=0, end=epochs,
                               start_factor=1., end_factor=0.1)]
    cfg.default_hooks.param_scheduler = dict(type='ParamSchedulerHook')
    cfg.default_hooks.checkpoint = dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=2,
        save_best='voc_standard/mAP', rule='greater', save_optimizer=True)
    # No EMA/PipelineSwitch: only adapters learn, original buffers must stay fixed.
    cfg.custom_hooks = [dict(type='VOCAdapterAuditHook'),
                        dict(type='BDDResumePathsHook'),
                        dict(type='BDDSessionBudgetHook', deadline_unix=deadline_unix)]
    contract = dict(schema=1, case_id=case_id, case=case,
                    train_manifest_sha256=train['manifest_sha256'] if train else None,
                    train_count=train['count'] if train else None,
                    validation_count=val['count'],
                    val_manifest_sha256=val['manifest_sha256'],
                    epochs=epochs, batch_per_gpu=batch, world_size=num_gpus,
                    workers=workers, seed=seed, lr=lr, reduction=reduction,
                    optimizer=dict(cfg.optim_wrapper), scheduler=list(cfg.param_scheduler),
                    model=dict(cfg.model), gamma=[1.5, 5.0],
                    selection='voc_standard/mAP on fixed VOC_Dark validation',
                    precision='fp32', ema=False)
    cfg.adapter_contract = contract
    return cfg


def canonical_contract(contract):
    return json.loads(json.dumps(contract, sort_keys=True))


def prepare_weights(repo, cfg, source, destination, checkpoint_id):
    """Validate every model tensor; keep VOC classifiers, reject unknown changes."""
    import torch
    from mmyolo.registry import MODELS
    from mmyolo.utils import register_all_modules
    from mmengine.runner import set_random_seed
    register_all_modules()
    import projects.lowlight_adapter  # noqa: F401
    set_random_seed(cfg.randomness['seed'])
    source = Path(source)
    original = shared.load_torch_checkpoint(source)
    model = MODELS.build(cfg.model)
    model.init_weights()
    state = {k.removeprefix('module.'): v for k, v in original['state_dict'].items()}
    expected = model.state_dict()
    removed = sorted(k for k in state if k.startswith('criterion_lpips.'))
    state = {k: v for k, v in state.items() if k not in removed}
    unexpected = set(state) - set(expected)
    missing = set(expected) - set(state)
    allowed_missing = {k for k in expected if k.startswith('adapters.')} if checkpoint_id != 2 else set()
    if unexpected or missing != allowed_missing:
        raise ValueError(f'Checkpoint contract mismatch: missing={sorted(missing)}, '
                         f'unexpected={sorted(unexpected)}')
    mismatch = [k for k in state if state[k].shape != expected[k].shape]
    if mismatch:
        raise ValueError(f'Checkpoint tensor shapes differ: {mismatch}')
    model.load_state_dict(state, strict=not missing)
    saved_meta = original.get('meta', {}).get('voc_adapter', {})
    report = dict(source=str(source), source_sha256=shared.sha256(source),
                  base_checkpoint_sha256=saved_meta.get('base_checkpoint_sha256', shared.sha256(source)),
                  source_checkpoint_id=checkpoint_id, case_id=cfg.adapter_contract['case_id'],
                  source_epoch=original.get('meta', {}).get('epoch'),
                  removed_lpips_keys=removed, initialized_adapter_keys=sorted(missing),
                  classifier_policy='Keep both complete VOC classifiers unchanged',
                  weight_field='state_dict; no selection between EMA/original using validation')
    if saved_meta.get('training_contract'):
        report['training_contract'] = saved_meta['training_contract']
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Save complete state, including identity adapters, to allow strict loading later.
    torch.save(dict(state_dict=model.state_dict(), meta=dict(
        dataset_meta=dict(classes=VOC_CLASSES), voc_adapter=report)), destination)
    shared.write_json(destination.with_suffix('.json'), report)
    cfg.adapter_contract['base_checkpoint_sha256'] = report['base_checkpoint_sha256']
    cfg.adapter_source_training_contract = saved_meta.get('training_contract')
    cfg.adapter_source_epoch = report['source_epoch']
    cfg.adapter_identity_expected = checkpoint_id != 2
    del model
    return destination


def validate_resume(cfg, source):
    saved = shared.load_torch_checkpoint(source)
    meta = saved.get('meta', {}).get('voc_adapter', {})
    contract = meta.get('training_contract')
    if not contract or 'optimizer' not in saved:
        raise ValueError('Resume needs notebook-05 epoch_N.pth with training contract/optimizer')
    cfg.adapter_contract['base_checkpoint_sha256'] = meta['base_checkpoint_sha256']
    if canonical_contract(contract) != canonical_contract(cfg.adapter_contract):
        raise ValueError('Resume contract differs: preserve case, base checkpoint, manifests, '
                         'total epochs, batch, LR, reduction, seed and GPU count')
    if saved.get('meta', {}).get('epoch', 0) >= cfg.max_epochs:
        raise ValueError('Training is already complete at the requested total epoch count')
    shared.copy_resume_best(source, cfg.work_dir)
    cfg.load_from, cfg.resume = str(source), True


def run(repo, cfg, checkpoint, action, num_gpus=2, benchmark_images=16):
    if action == 'train':
        if not CASES[cfg.adapter_contract['case_id']]['train']:
            raise ValueError('No trainable adapter in cases 0/1')
        cfg.load_from = str(checkpoint)
    folder = Path(cfg.work_dir)
    folder.mkdir(parents=True, exist_ok=True)
    config_file = folder / 'resolved_config.py'
    cfg.dump(str(config_file))
    record = dict(action=action, source_checkpoint=str(checkpoint),
                  checkpoint_sha256=shared.sha256(checkpoint),
                  workflow_sha256=shared.sha256(__file__),
                  config_sha256=shared.sha256(config_file), contract=cfg.adapter_contract,
                  repo_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                                      text=True).strip(),
                  repo_status=subprocess.check_output(['git', 'status', '--short'], cwd=repo,
                                                      text=True),
                  dependency_versions={p: shared.version(p) for p in (
                      'torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'mmyolo')},
                  final_test=False)
    record['source_files_sha256'] = {str(p.relative_to(repo)): shared.sha256(p)
                                    for p in (Path(repo) / 'projects/lowlight_adapter').glob('*.py')}
    import inspect
    from mmdet.datasets import VOCDataset
    from mmdet.evaluation.metrics import VOCMetric
    record['installed_evaluation_sources'] = {}
    for name, obj in [('voc_metric', VOCMetric), ('voc_xml_parser', VOCDataset.parse_data_info)]:
        actual = Path(inspect.getsourcefile(obj))
        snapshot = folder / f'installed_{name}.py'
        snapshot.write_text(actual.read_text())
        record['installed_evaluation_sources'][name] = dict(
            import_path=str(actual), sha256=shared.sha256(actual), snapshot=str(snapshot))
    record['final_test'] = bool(cfg.get('adapter_selection_locked', False)
                                and cfg.get('adapter_evaluation_domain') == 'exdark')
    (folder / 'source_diff.patch').write_text(subprocess.check_output(
        ['git', 'diff', '--binary'], cwd=repo, text=True))
    shared.write_json(folder / 'provenance.json', record)
    if action == 'train':
        command = shared.training_command(config_file, num_gpus)
    else:
        command = [sys.executable, '-m', 'projects.lowlight_adapter.run',
                   str(config_file), str(checkpoint), '--action', action,
                   '--benchmark-images', str(benchmark_images)]
    log = folder / f'{action}.log'
    with log.open('w') as handle:
        try:
            if action == 'train':
                print('Running:', ' '.join(map(str, command)), flush=True)
                with subprocess.Popen(list(map(str, command)), cwd=repo,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, bufsize=1) as process:
                    for line in process.stdout:
                        handle.write(line)
                        handle.flush()
                        if any(key in line for key in ('Epoch(train)', 'Epoch(val)',
                                                      'checkpoint', 'ERROR', 'Traceback')):
                            print(line.rstrip(), flush=True)
                    if process.wait():
                        raise subprocess.CalledProcessError(process.returncode, command)
            else:
                shared.call(command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            tail = '\n'.join(log.read_text(errors='replace').splitlines()[-100:])
            raise RuntimeError(f'{action} failed; inspect {log}\n{tail}') from exc
    print('Log:', log)
    if action != 'train':
        return json.loads((folder / f'{action}_report.json').read_text())
    state_file = folder / 'session_state.json'
    status = json.loads(state_file.read_text()) if state_file.exists() else {}
    if status.get('status') != 'completed':
        print('Session incomplete; preserve the entire stage folder and resume.')
        return status
    best = sorted(folder.glob('best_voc_standard_mAP_epoch_*.pth'))
    if len(best) != 1:
        raise RuntimeError(f'Expected one best validation checkpoint: {best}')
    status['selected_checkpoint'] = str(best[0])
    shared.write_json(folder / 'selection.json', status)
    return status


def execute(repo, output, run_id, checkpoint_id, source, case_id, action, mode,
            datasets, download_release=False, epochs=None, batch=4, workers=2,
            seed=42, lr=0.001, num_gpus=2, reduction=16, deadline_unix=None,
            resume=False, final_evaluation=False, benchmark_images=32):
    """Run exactly one selected case. Tests are never launched after training."""
    chosen = selection(checkpoint_id, case_id, action)
    if mode not in ('pilot', 'full'):
        raise ValueError('MODE must be pilot or full')
    if action == 'evaluate' and mode != 'full':
        raise ValueError('Use MODE=full for evaluation; pilot is only for smoke/train')
    if resume and (action != 'train' or checkpoint_id != 2):
        raise ValueError('Resume requires ACTION=train, CHECKPOINT_ID=2 and an epoch checkpoint')
    if action == 'train' and checkpoint_id == 2 and not resume:
        raise ValueError('Use RESUME=True to continue an adapted epoch; fresh cases start from 0/1')
    folder = Path(output) / run_id
    if folder.exists():
        raise FileExistsError('Choose a new RUN_ID; old artifacts are preserved')
    val = dataset_spec(domain='voc', **datasets['voc'])
    if val['count'] != 3760:
        raise ValueError('Expected released VOC_Dark_Val manifest with 3,760 IDs')
    train = None
    if action in ('audit', 'smoke', 'train') and chosen['train']:
        train = dataset_spec(domain='train', **datasets['train'])
        if train['count'] != 12334:
            raise ValueError('Expected released VOC training manifest with 12,334 IDs')
        assert_no_train_val_overlap(train, val)
    audits = dict(voc=val, train=train)
    if final_evaluation:
        if action != 'evaluate':
            raise ValueError('FINAL_EVALUATION is only meaningful for ACTION=evaluate')
        exdark = dataset_spec(domain='exdark', **datasets['exdark'])
        if exdark['count'] != 2563:
            raise ValueError('Use the author ExDark subset manifest (2,563 images), not full ExDark')
        audits['exdark'] = exdark
    shared.write_json(folder / 'dataset_audit.json', audits)
    shared.write_json(folder / 'case.json', chosen)
    if action == 'audit':
        print(json.dumps(audits, indent=2))
        return audits
    source = resolve_checkpoint(checkpoint_id, source, output, case_id, download_release)
    if train is not None and mode == 'pilot':
        train = subset_spec(train, 1000, folder / 'train_ids.txt', seed)
        val = subset_spec(val, 200, folder / 'val_ids.txt', seed)
    epochs = (3 if mode == 'pilot' else 5) if epochs is None else epochs
    cfg = build_config(repo, case_id, folder / 'training' if action == 'train' else folder / action,
                       val, train, epochs=epochs, batch=batch, workers=workers, seed=seed,
                       lr=lr, num_gpus=num_gpus, reduction=reduction,
                       deadline_unix=deadline_unix)
    compatible = prepare_weights(repo, cfg, source, folder / 'initialization/compatible.pth', checkpoint_id)
    shared.write_json(folder / 'experiment.json', dict(
        mode=mode, chosen=chosen, contract=cfg.adapter_contract,
        original_checkpoint=str(source), original_checkpoint_sha256=shared.sha256(source),
        pilot_is_scientific_result=False, tests_used_for_selection=False))
    if action == 'evaluate':
        if final_evaluation:
            source_training = cfg.get('adapter_source_training_contract')
            if source_training and source_training.get('train_count') != 12334:
                raise ValueError('Pilot-adapted checkpoints cannot be used for final result rows')
            # Declaration saved BEFORE any ExDark metric is computed.
            shared.write_json(folder / 'selection_locked_before_test.json', dict(
                case_id=case_id, source_sha256=shared.sha256(source),
                declaration='Model/hyperparameters selected with VOC validation only'))
        reports = []
        for spec in ([val, audits['exdark']] if final_evaluation else [val]):
            evaluation_cfg = build_config(
                repo, case_id, folder / spec['domain'], spec, workers=workers,
                seed=seed, num_gpus=1, reduction=reduction)
            evaluation_cfg.adapter_source_training_contract = cfg.get('adapter_source_training_contract')
            evaluation_cfg.adapter_source_epoch = cfg.get('adapter_source_epoch')
            evaluation_cfg.adapter_contract['base_checkpoint_sha256'] = cfg.adapter_contract['base_checkpoint_sha256']
            evaluation_cfg.adapter_selection_locked = final_evaluation
            evaluation_cfg.adapter_evaluation_domain = spec['domain']
            report = run(repo, evaluation_cfg, compatible, 'evaluate', num_gpus=1,
                         benchmark_images=benchmark_images)
            reports.append(str(folder / spec['domain'] / 'evaluate_report.json'))
        shared.write_json(folder / 'evaluation_reports.json', reports)
        return dict(reports=reports)
    if action == 'train':
        # One real backward check in a separate process; its update is discarded.
        smoke_cfg = copy.deepcopy(cfg)
        smoke_cfg.work_dir = str(folder / 'pretrain_smoke')
        run(repo, smoke_cfg, compatible, 'smoke', num_gpus=1, benchmark_images=0)
        if resume:
            validate_resume(cfg, source)
            import torch
            normalized = folder / 'training/resume_compatible.pth'
            normalized.parent.mkdir(parents=True, exist_ok=True)
            torch.save(shared.load_torch_checkpoint(source), normalized)
            cfg.load_from = str(normalized)
            compatible = normalized
        return run(repo, cfg, compatible, 'train', num_gpus=num_gpus)
    return run(repo, cfg, compatible, 'smoke', num_gpus=1, benchmark_images=0)


def export_tables(reference_csv, summaries, output, proposed_case_id=3):
    """Never invent metrics or infer that a protocol is paper-equivalent."""
    with Path(reference_csv).open() as handle:
        rows = list(csv.DictReader(handle))
    if proposed_case_id not in (2, 3, 4):
        raise ValueError('Select one proposed model case 2/3/4 before inspecting final tests')
    own, efficiency = {}, []
    bases, manifest_by_domain = set(), {}
    for summary in summaries:
        report = json.loads(Path(summary).read_text())
        if report.get('scope') != 'full_evaluation':
            raise ValueError('Only full evaluation reports may enter result tables')
        if report['domain'] == 'exdark' and not report.get('selected_before_test'):
            raise ValueError('ExDark result lacks a pre-test selection-lock declaration')
        bases.add(report['base_checkpoint_sha256'])
        if len(bases) > 1:
            raise ValueError('Use separate comparison tables for different source checkpoints')
        domain = report['domain']
        manifest = report['evaluation_manifest_sha256']
        if domain in manifest_by_domain and manifest_by_domain[domain] != manifest:
            raise ValueError('Compared methods must use the same evaluation manifest')
        manifest_by_domain[domain] = manifest
        key = (report['case_id'], report['model_sha256'])
        row = own.setdefault(key, dict(method='LR-YOLOv10 + ' + CASES[key[0]]['name'], case_id=key[0],
            source='measured', voc_map='', voc_all_objects='', exdark_map='',
            protocol='local VOC 11-point AP50; all-object equivalence unverified',
            checkpoint_sha256=report['checkpoint_sha256'], model_sha256=key[1]))
        if report['domain'] == 'voc':
            row['voc_map'] = 100 * report['metrics']['voc_standard/mAP']
            row['voc_all_objects'] = 100 * report['metrics']['voc_all_objects/mAP']
        elif report['domain'] == 'exdark':
            row['exdark_map'] = 100 * report['metrics']['voc_standard/mAP']
        efficiency.append(dict(case_id=key[0], domain=report['domain'], model_sha256=key[1],
                               **(report.get('parameters') or {}), **(report.get('latency') or {})))
    # Group by model tensors; metadata-only compatibility copies have different file hashes.
    if any(k[0] == proposed_case_id for k in own) and not any(k[0] == 0 for k in own):
        raise ValueError('Include a measured case-0 baseline with the proposed result')
    rows.extend(own.values())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ['method', 'source', 'case_id', 'voc_map', 'voc_all_objects', 'exdark_map',
              'protocol', 'checkpoint_sha256', 'model_sha256']
    with (output / 'table1_lowlight_extended.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'table1_main.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(r for r in rows if r['source'] == 'paper_table1'
                         or r['case_id'] in (0, proposed_case_id))
    with (output / 'adapter_ablation.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(own.values())
    shared.write_json(output / 'efficiency.json', efficiency)
    if efficiency:
        columns = sorted(set().union(*(r.keys() for r in efficiency)))
        with (output / 'efficiency.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(efficiency)
    return rows
