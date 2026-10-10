_base_ = './1_yolov8_s_voc_fog_100e_haze_train_no_aug.py'

# Parameter-space smoothing using two first-order backward passes.
model = dict(
    type='YOLODetectorSAMNorm',
    pgn_config=dict(enable=True, rho=0.05, alpha=0.8, adaptive=False))
load_from = None
