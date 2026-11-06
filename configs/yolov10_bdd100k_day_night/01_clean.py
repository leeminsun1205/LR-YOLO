"""BDD clean detection + reconstruction stage (all detector parameters learn).

Architecture, augmentations and L1/LPIPS come from the released VOC config.
Default notebook initialization is scratch, as in the original clean config.
Optional VOC transfer is a separately labeled experiment.
"""
_base_ = ['../yolov10_voc_dark/2_yolov10_s_reconstruct_voc_dark_100e_no_aug.py']

custom_imports = dict(imports=['projects.bdd100k.dataset', 'projects.bdd100k.hooks'],
                      allow_failed_imports=False)
dataset_type = 'YOLOv5BDD100KDayNightDataset'
data_root = ('/kaggle/input/datasets/lmnhnht125/'
             'bdd100k-day-night-voc-lryolo/bdd100k_day_night_voc/')
classes = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
           'motorcycle', 'bicycle', 'traffic light', 'traffic sign')
num_classes = 10
train_batch_size_per_gpu = 2
train_num_workers = 2
val_num_workers = 2
max_epochs = 100
base_lr = 0.01
num_gpus = 2
launcher = 'pytorch'

train_dataloader = dict(
    batch_size=2, num_workers=2, persistent_workers=True,
    dataset=dict(type=dataset_type, data_root=data_root,
                 metainfo=dict(classes=classes),
                 ann_file='train/ImageSets/Main/train.txt',
                 data_prefix=dict(sub_data_root='train/')))
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    dataset=dict(type=dataset_type, data_root=data_root,
                 metainfo=dict(classes=classes),
                 ann_file='val/ImageSets/Main/val.txt',
                 data_prefix=dict(sub_data_root='val/')))
# Never expose final test domains through a training config's test loop.
test_dataloader = val_dataloader
optim_wrapper = dict(optimizer=dict(lr=base_lr, batch_size_per_gpu=2))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100,
                 val_interval=1, dynamic_intervals=[])
default_hooks = dict(
    param_scheduler=dict(max_epochs=100),
    checkpoint=dict(interval=1, save_best='pascal_voc/mAP', rule='greater',
                    max_keep_ckpts=2, save_optimizer=True,
                    save_param_scheduler=True, save_last=True))
# Original stage-2 switch pipeline is identical to the first pipeline.
custom_hooks = [dict(type='EMAHook', ema_type='ExpMomentumEMA',
                     momentum=0.0001, update_buffers=True,
                     strict_load=False, priority=49),
                dict(type='BDDRunDiagnosticsHook'),
                dict(type='BDDResumePathsHook'),
                dict(type='BDDSessionBudgetHook')]
val_evaluator = dict(type='mmdet.VOCMetric', metric='mAP',
                     iou_thrs=0.5, eval_mode='11points')
test_evaluator = val_evaluator
randomness = dict(seed=42, deterministic=False)
env_cfg = dict(cudnn_benchmark=False)
load_from = None
resume = False
work_dir = '/kaggle/working/lryolo_bdd100k/clean'
