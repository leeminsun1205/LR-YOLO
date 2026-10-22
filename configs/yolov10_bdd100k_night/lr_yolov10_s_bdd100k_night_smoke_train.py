_base_ = ['./lr_yolov10_s_bdd100k_night.py']

max_epochs = 1
save_epoch_intervals = 1
val_interval_stage2 = 1

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,
    val_interval=1
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto'
    )
)

work_dir = 'work_dirs/lr_yolov10_s_bdd100k_night_smoke_train'