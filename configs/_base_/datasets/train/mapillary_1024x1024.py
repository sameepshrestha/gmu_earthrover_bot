# dataset settings
dataset_type_train = 'MapillaryVistasDataset'
data_root_train = 'data/mapillary/'
crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(2048, 1024)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4, # This is overwritten in the final config
    num_workers=1, # This is overwritten in the final config
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type_train,
        data_root=data_root_train,
        data_prefix=dict(
            img_path='training/images', seg_map_path='training/labels'),
        pipeline=train_pipeline))