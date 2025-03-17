_base_ = [
    '_base_/models/eva-clip+mask2former.py',
    '_base_/datasets/train/cityscapes_1024x1024.py',
    '_base_/datasets/test/synth2cityscapes.py',
    '_base_/schedules/schedule_1k_frozen.py',
    '_base_/default_runtime.py'
]

crop_size = (1024, 1024)
stride_size = (768,768)

# crop_size = (512, 512)
# stride_size = (426,426)

num_gpus = 2
num_samples_per_gpu_train = 8
num_workers_per_gpu_train = 1
num_samples_per_gpu_test = 2
num_workers_per_gpu_test = 1

# model = dict(
#     data_preprocessor=dict(size=crop_size),
#     backbone=dict(
#         img_size=crop_size[0]
#     ),
#     test_cfg=dict(mode='slide', crop_size=crop_size, stride=stride_size)
# )

model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(img_size=crop_size[0]),
    decode_head=dict(num_classes=19),  # Ensure 19 classes
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=stride_size)
)

train_dataloader = dict(
    batch_size=num_samples_per_gpu_train,
    num_workers=num_workers_per_gpu_train,
)

val_dataloader = dict(
    batch_size=num_samples_per_gpu_test,
    num_workers=num_workers_per_gpu_test,
)

test_dataloader = dict(
    batch_size=num_samples_per_gpu_test,
    num_workers=num_workers_per_gpu_test,
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (# GPUs) x (# samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=num_gpus*num_samples_per_gpu_train)