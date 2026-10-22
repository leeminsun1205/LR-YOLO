_base_ = ['./lr_yolov10_s_bdd100k_night.py']

max_epochs = 1

train_batch_size_per_gpu = 2

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        ann_file='/kaggle/working/train_small.txt'
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file='/kaggle/working/val_small.txt'
    )
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,
    val_interval=1
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1
    )
)

work_dir = 'work_dirs/lr_yolov10_s_bdd100k_night_smoke_train'