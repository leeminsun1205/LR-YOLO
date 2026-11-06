"""Original frozen-decoder adaptation, initialized from the BDD clean stage.

Training and internal validation both sample gamma online, as in the released
training implementation. The daytime input itself supplies clean gts.
"""
_base_ = ['./01_clean.py']

model = dict(
    type='YOLOReconstructDehazeDetector',
    data_preprocessor=dict(type='YOLOv5DarkDetDataPreprocessor'))
load_from = '/kaggle/working/lryolo_bdd100k/clean/epoch_100.pth'
work_dir = '/kaggle/working/lryolo_bdd100k/dark'
