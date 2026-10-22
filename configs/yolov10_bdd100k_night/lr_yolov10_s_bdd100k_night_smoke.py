_base_ = ['./lr_yolov10_s_bdd100k_night.py']

# Dùng danh sách test nhỏ tạo trong Kaggle notebook
test_dataloader = dict(
    dataset=dict(
        ann_file='/kaggle/working/test_small.txt'
    )
)

work_dir = 'work_dirs/lr_yolov10_s_bdd100k_night_smoke'