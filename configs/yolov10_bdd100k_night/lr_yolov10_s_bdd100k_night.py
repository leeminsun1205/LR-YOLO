_base_ = [
    '../yolov10_voc_dark/'
    '3_finetune_yolov10_s_reconstruct_voc_dark_ep100_dark_train_no_aug_dark_test.py'
]

dataset_type = 'YOLOv5BDD100KNightDataset'

data_root = (
    '/kaggle/input/datasets/lmnhnht125/'
    'bdd100k-night-voc-lryolo/'
    'bdd100k_night_voc/'
)

train_ann_file = 'train/ImageSets/Main/train.txt'
val_ann_file = 'val/ImageSets/Main/val.txt'
test_ann_file = 'test/ImageSets/Main/test.txt'

train_data_prefix = 'train/'
val_data_prefix = 'val/'
test_data_prefix = 'test/'

num_classes = 10

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(sub_data_root=train_data_prefix),
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(sub_data_root=val_data_prefix),
    )
)

test_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=test_ann_file,
        data_prefix=dict(sub_data_root=test_data_prefix),
    )
)

load_from = (
    '/kaggle/working/LR-YOLO/checkpoints/'
    'lr_yolov10_voc_dark_compatible.pth'
)

work_dir = 'work_dirs/lr_yolov10_s_bdd100k_night'