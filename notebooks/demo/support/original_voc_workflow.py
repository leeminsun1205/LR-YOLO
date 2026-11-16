"""One-epoch timing of the released clean/dark VOC configs, not adapter training.

No runtime work at import. Called explicitly by notebook 06 on Kaggle only.
"""
import bisect
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

import bdd100k_workflow as shared
import voc_adapter_workflow as adapters

SOURCE_CONFIGS = {
    'clean': 'configs/yolov10_voc_dark/2_yolov10_s_reconstruct_voc_dark_100e_no_aug.py',
    'dark': 'configs/yolov10_voc_dark/3_finetune_yolov10_s_reconstruct_voc_dark_ep100_dark_train_no_aug.py',
}
MODEL_CONFIGS = dict(v10=SOURCE_CONFIGS, v8={
    'clean': 'configs/yolov8_voc_dark/2_yolov8_s_reconstruct_voc_dark_100e_no_aug.py',
    'dark': 'configs/yolov8_voc_dark/3_finetune_yolov8_s_reconstruct_voc_dark_ep100_dark_train_no_aug.py',
})
DEGRADATION_CONFIGS = {'low_light': MODEL_CONFIGS, 'haze': {
    version: {
        'clean': f'configs/yolo{version}_voc_fog/2_yolo{version}_s_reconstruct_voc_fog_100e_no_aug.py',
        'dark': f'configs/yolo{version}_voc_fog/3_finetune_yolo{version}_s_reconstruct_voc_fog_100e_haze_train_no_aug.py',
    } for version in ('v8','v10')}}


def prepare_data(overrides, cache, auto_download=True, archives=None, search_roots=None):
    """Original train configs need CLEAN val from the VOC training bundle."""
    prepared = adapters.prepare_datasets(
        dict(train=overrides.get('train', {})), cache, 3, 'train',
        auto_download=auto_download, archives=archives, search_roots=search_roots,
        domains=('train',))
    train_settings = prepared['train']
    val_settings = overrides.get('clean_val', {})
    if val_settings.get('root') is None:
        train_folder = Path(train_settings['root']) / train_settings['prefix']
        val_settings = adapters.discover_dataset(train_folder.parent, 'voc')
        if val_settings is None:
            raise FileNotFoundError(
                f'No CLEAN validation manifest with 3,760 IDs found beside {train_folder}. '
                'Set CLEAN_VAL_ROOT/MANIFEST/PREFIX to the original clean validation. '
                'Do not use fixed offline VOC_Dark_Val: dark stage would darken it twice.')
    folder = (Path(val_settings['root']) / val_settings['prefix']).resolve()
    if any(p.lower().replace('-', '_') in ('voc_dark_test', 'voc_dark_val') for p in folder.parts):
        raise ValueError('Original training benchmark requires clean validation, '
                         'not the fixed offline VOC_Dark_Val package')
    train = adapters.dataset_spec(domain='train', **train_settings)
    val = adapters.dataset_spec(domain='voc', **val_settings)
    if train['count'] != 12334 or val['count'] != 3760:
        raise ValueError('Expected full original 12,334 train / 3,760 clean validation')
    adapters.assert_no_train_val_overlap(train, val)
    return dict(train=train, clean_val=val,
                protocol='clean validation for clean stage; original online gamma for dark validation')


def build_config(repo, stage, datasets, work_dir, clean_checkpoint=None,
                 batch=4, workers=2, seed=42, num_gpus=2, warmup_steps=20,
                 model_version='v10', degradation='low_light'):
    from mmengine.config import Config
    if (degradation not in DEGRADATION_CONFIGS or model_version not in MODEL_CONFIGS or stage not in SOURCE_CONFIGS
            or batch < 1 or workers < 0 or num_gpus not in (1, 2)):
        raise ValueError('Invalid stage/batch/workers/GPU settings')
    if stage == 'dark' and (clean_checkpoint is None or not Path(clean_checkpoint).is_file()):
        raise FileNotFoundError('Dark timing needs the completed clean probe epoch_1.pth '
                                'or an explicit trusted full clean-stage checkpoint')
    source_config = DEGRADATION_CONFIGS[degradation][model_version][stage]
    cfg = Config.fromfile(str(Path(repo) / source_config))
    original_loop = copy.deepcopy(dict(cfg.train_cfg))
    original_checkpoint_hook = copy.deepcopy(dict(cfg.default_hooks.checkpoint))
    if original_loop['max_epochs'] != 100 or cfg.default_hooks.param_scheduler.max_epochs != 100:
        raise ValueError('Expected the inspected original 100-epoch config/scheduler')
    expected_type = 'YOLOReconstructDetector' if stage == 'clean' else 'YOLOReconstructDehazeDetector'
    expected_preprocessor = ('YOLOv5DetDataPreprocessor' if stage == 'clean' else
        ('YOLOv5DarkDetDataPreprocessor' if degradation == 'low_light' else 'YOLOv5HazeDetDataPreprocessor'))
    if cfg.model.type != expected_type or cfg.model.data_preprocessor.type != expected_preprocessor:
        raise ValueError('Unexpected original detector/preprocessing contract')
    if cfg.model.loss_l1_weight != 100 or cfg.model.loss_lpips_weight != 100:
        raise ValueError('Do not silently change original reconstruction loss weights')
    cfg.custom_imports = dict(imports=['mmyolo.datasets.yolov5_voc_dark' if degradation == 'low_light'
                                      else 'mmyolo.datasets.yolov5_voc_fog',
                                      'projects.original_runtime'], allow_failed_imports=False)
    for name, spec in [('train', datasets['train']), ('val', datasets['clean_val'])]:
        loader = cfg[name + '_dataloader']
        loader.dataset.update(data_root=spec['root'] + '/', ann_file=spec['manifest'],
                              data_prefix=dict(sub_data_root=spec['prefix']))
        loader.update(num_workers=workers, persistent_workers=workers > 0)
    cfg.train_dataloader.batch_size = batch
    cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)
    cfg.train_batch_size_per_gpu = batch
    cfg.train_num_workers = cfg.val_num_workers = workers
    cfg.optim_wrapper.optimizer.batch_size_per_gpu = batch
    # Keep original model, optimizer LR/momentum/WD/clip, EMA and 100-epoch
    # scheduler/warmup. Only the executed loop and measurement frequency change.
    cfg.train_cfg.update(max_epochs=1, val_interval=1, dynamic_intervals=[])
    cfg.default_hooks.checkpoint.update(interval=1, max_keep_ckpts=1)
    cfg.custom_hooks += [dict(type='OriginalEpochTimingHook', warmup_steps=warmup_steps),
                         dict(type='OriginalEpochTimingTailHook')]
    cfg.load_from = str(clean_checkpoint) if stage == 'dark' else None
    cfg.resume = False
    cfg.randomness = dict(seed=seed, deterministic=False)
    cfg.work_dir = str(work_dir)
    cfg.launcher = 'pytorch' if num_gpus == 2 else 'none'
    cfg.original_runtime = dict(
        stage=stage, scientific_result=False, source_config=source_config,
        source_train_cfg=original_loop, source_checkpoint_hook=original_checkpoint_hook,
        full_train_count=datasets['train']['count'], full_val_count=datasets['clean_val']['count'],
        train_manifest_sha256=datasets['train']['manifest_sha256'],
        clean_val_manifest_sha256=datasets['clean_val']['manifest_sha256'],
        initialization='scratch' if stage == 'clean' else 'supplied clean checkpoint; '
                       'default is timing-only clean epoch 1, not original epoch 100',
        hardware_overrides=dict(batch_per_gpu=batch, world_size=num_gpus,
                                global_batch=batch * num_gpus, workers_per_rank=workers),
        changes=['one epoch loop', 'force one validation and regular checkpoint',
                 'bounded checkpoint retention', 'seed and hardware settings', 'timing hooks'],
        preserved=['original detector/backbone/neck/head', 'L1/LPIPS 100/100',
                   'original gamma and validation preprocessing', 'original freeze policy including BN behavior',
                   'SGD LR 0.01/momentum 0.937/WD 0.0005/Nesterov',
                   'original EMA, augmentation, gradient clipping and 100-epoch scheduler'])
    return cfg


def validation_epochs(source_loop):
    """Pinned MMEngine 0.10.7 completed_epoch+1 milestone semantics."""
    dynamic = source_loop.get('dynamic_intervals', [])
    milestones = [0] + [int(v[0]) for v in dynamic]
    intervals = [source_loop['val_interval']] + [int(v[1]) for v in dynamic]
    epochs = source_loop['max_epochs']
    result = []
    for completed in range(1, epochs + 1):
        interval = intervals[bisect.bisect_right(milestones, completed + 1) - 1]
        if completed >= source_loop.get('val_begin', 1) and (
                completed % interval == 0 or completed == epochs):
            result.append(completed)
    return result


def estimate_stage(ranks, source_loop, checkpoint_hook, process_wall_seconds):
    """Timing-based extrapolation, not a bound or training/convergence result."""
    if not ranks or any(r['completed_epochs'] != 1 for r in ranks):
        raise ValueError('Need one complete original epoch per rank')
    schedule = validation_epochs(source_loop)
    if any(r['original_validation_epochs'] != schedule for r in ranks):
        raise ValueError('Installed loop validation schedule differs from the pinned estimate')
    epochs = source_loop['max_epochs']
    interval = checkpoint_hook.get('interval', 1)
    checkpoint_epochs = [e for e in range(1, epochs + 1)
                         if e % interval == 0 or (e == epochs and checkpoint_hook.get('save_last', True))]
    train = max(r['train_epoch_seconds'] for r in ranks)
    val = max(r['val_epoch_seconds'] for r in ranks)
    regular = max(r['regular_checkpoint_seconds'] for r in ranks)
    best = max(r['best_checkpoint_seconds'] for r in ranks)
    overhead = max(0., process_wall_seconds - train - val - regular - best)
    steady_values = [r['steady_epoch_seconds_estimate'] for r in ranks]
    steady = max(steady_values) if all(v is not None for v in steady_values) else None
    base = epochs * train + len(schedule) * val + len(checkpoint_epochs) * regular + overhead
    return dict(
        target_epochs=epochs, train_epoch_seconds=train, validation_seconds_per_run=val,
        source_validation_epochs=schedule, validation_runs=len(schedule),
        source_regular_checkpoint_epochs=checkpoint_epochs,
        startup_shutdown_other_seconds=overhead,
        projection_seconds_excluding_future_best_saves=base,
        projection_seconds_if_best_saved_every_validation=base + len(schedule) * best,
        projection_hours_excluding_future_best_saves=base / 3600,
        projection_hours_if_best_saved_every_validation=(base + len(schedule) * best) / 3600,
        steady_projection_hours_excluding_future_best_saves=(
            (epochs * steady + len(schedule) * val + len(checkpoint_epochs) * regular + overhead) / 3600
            if steady is not None else None),
        observed_best_save_gap_seconds=best,
        max_rank_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in ranks),
        max_rank_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in ranks),
        caution='First-epoch estimate; warm caches, later prediction density, LR/warmup, '
                'Kaggle contention and checkpoint frequency can change runtime. '
                'Best-save endpoints are workload scenarios, not confidence bounds.')


def run_stage(repo, cfg, num_gpus):
    folder = Path(cfg.work_dir)
    if folder.exists():
        raise FileExistsError(f'Use a fresh runtime run directory: {folder}')
    folder.mkdir(parents=True)
    config_file = folder / 'resolved_config.py'
    cfg.dump(str(config_file))
    from mmengine.runner.loops import EpochBasedTrainLoop
    from mmengine.runner.utils import calc_dynamic_intervals
    for name, obj in [('installed_train_loop.py', EpochBasedTrainLoop),
                      ('installed_loop_utils.py', calc_dynamic_intervals)]:
        path = Path(inspect.getsourcefile(obj))
        (folder / name).write_text(path.read_text())
    provenance = dict(
        probe=cfg.original_runtime, resolved_config_sha256=shared.sha256(config_file),
        source_config_sha256=shared.sha256(Path(repo) / cfg.original_runtime['source_config']),
        initialization_checkpoint=cfg.load_from,
        initialization_sha256=shared.sha256(cfg.load_from) if cfg.load_from else None,
        repo_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        versions={p: shared.version(p) for p in ('torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'mmyolo', 'lpips')},
        workflow_sha256=shared.sha256(__file__))
    paths = list((Path(repo) / 'projects/original_runtime').glob('*.py'))
    paths += [Path(repo) / p for p in (
        'mmyolo/models/detectors/yolo_reconstruct_dehaze_detector.py',
        'mmyolo/models/backbones/yolov10_backbone_reconstruct.py',
        'mmyolo/models/dense_heads/yolov10_head.py',
        'mmyolo/models/data_preprocessors/dark_data_preprocessor.py',
        'mmyolo/models/data_preprocessors/data_preprocessor.py')]
    provenance['source_files_sha256'] = {str(p.relative_to(repo)): shared.sha256(p) for p in paths}
    (folder / 'source_diff.patch').write_text(subprocess.check_output(
        ['git', 'diff', '--binary'], cwd=repo, text=True))
    shared.write_json(folder / 'provenance.json', provenance)
    command = shared.training_command(config_file, num_gpus)
    log = folder / 'train.log'
    print('Running:', ' '.join(map(str, command)), flush=True)
    started = time.perf_counter()
    with log.open('w') as handle, subprocess.Popen(
            list(map(str, command)), cwd=repo, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if any(key in line for key in ('Epoch(train)', 'Epoch(val)', 'Downloading',
                                           'checkpoint', 'ERROR', 'Traceback', 'Using SyncBatchNorm')):
                print(line.rstrip(), flush=True)
        code = process.wait()
    wall = time.perf_counter() - started
    if code:
        tail = '\n'.join(log.read_text(errors='replace').splitlines()[-100:])
        reason = 'CUDA OOM; choose a smaller batch and a NEW run ID' if (
            'out of memory' in tail.lower()) else 'Inspect the original traceback'
        raise RuntimeError(f'{reason}: {log}\n{tail}')
    ranks = [json.loads((folder / f'runtime_rank_{rank}.json').read_text()) for rank in range(num_gpus)]
    estimate = estimate_stage(ranks, cfg.original_runtime['source_train_cfg'],
                              cfg.original_runtime['source_checkpoint_hook'], wall)
    checkpoint = folder / 'epoch_1.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Original clean-to-dark handoff checkpoint missing: {checkpoint}')
    result = dict(stage=cfg.original_runtime['stage'], scientific_result=False,
                  process_wall_seconds=wall, checkpoint=str(checkpoint), ranks=ranks, estimate=estimate)
    shared.write_json(folder / 'runtime_summary.json', result)
    print('Stage report:', folder / 'runtime_summary.json', flush=True)
    return result


def benchmark(repo, datasets, output, stages=('clean', 'dark'), clean_checkpoint=None,
              batch=4, workers=2, seed=42, num_gpus=2, warmup_steps=20):
    if tuple(stages) not in (('clean',), ('dark',), ('clean', 'dark')):
        raise ValueError('STAGES must be clean, dark, or clean then dark')
    if tuple(stages) == ('dark',) and clean_checkpoint is None:
        raise ValueError('Dark-only probe needs CLEAN_STAGE_CHECKPOINT')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Fresh OUTPUT run directory required: {output}')
    shared.write_json(output / 'dataset_audit.json', datasets)
    result = dict(scientific_result=False, probe_epochs_per_stage=1, stages={})
    if clean_checkpoint is not None:
        # Preserve all model/LPIPS/classifier state; just normalize serialization.
        import torch
        normalized = output / 'initialization/clean_compatible.pth'
        normalized.parent.mkdir(parents=True)
        torch.save(shared.load_torch_checkpoint(clean_checkpoint), normalized)
        clean_checkpoint = normalized
    for stage in stages:
        cfg = build_config(repo, stage, datasets, output / stage,
                           clean_checkpoint=clean_checkpoint, batch=batch, workers=workers,
                           seed=seed, num_gpus=num_gpus, warmup_steps=warmup_steps)
        measured = run_stage(repo, cfg, num_gpus)
        result['stages'][stage] = measured
        if stage == 'clean' and clean_checkpoint is None:
            clean_checkpoint = measured['checkpoint']
        estimates = [v['estimate'] for v in result['stages'].values()]
        result['projection'] = dict(
            stages_measured=list(result['stages']), complete_two_stage_estimate=len(estimates) == 2,
            hours_excluding_future_best_saves=sum(e['projection_hours_excluding_future_best_saves'] for e in estimates),
            hours_if_best_saved_every_validation=sum(e['projection_hours_if_best_saved_every_validation'] for e in estimates),
            steady_hours_excluding_future_best_saves=(
                sum(e['steady_projection_hours_excluding_future_best_saves'] for e in estimates)
                if all(e['steady_projection_hours_excluding_future_best_saves'] is not None for e in estimates) else None),
            note='Each measured stage projected to its original 100 epochs. '
                 'Default dark initialization is clean epoch 1 for timing only, not clean epoch 100.')
        shared.write_json(output / 'summary.json', result)
    return result
