"""Final evaluation only. Notebook explicitly chooses one of three test splits."""
_base_ = ['./02_dark.py']

model = dict(data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'))
test_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=True,
    dataset=dict(ann_file='test_night/ImageSets/Main/test_night.txt',
                 data_prefix=dict(sub_data_root='test_night/')))
train_dataloader = None
train_cfg = None
val_dataloader = None
val_cfg = None
val_evaluator = None
load_from = None
resume = False
work_dir = '/kaggle/working/lryolo_bdd100k/evaluation'
