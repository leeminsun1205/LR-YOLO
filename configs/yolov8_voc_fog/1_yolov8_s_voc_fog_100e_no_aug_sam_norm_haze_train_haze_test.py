_base_ = './1_yolov8_s_voc_fog_100e_no_aug_sam_norm_haze_train.py'

model = dict(data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'))
val_dataloader = dict(dataset=dict(data_root='data/rtts/'))
test_dataloader = val_dataloader
