# dataset settings
dataset_type_test = 'ACDCDataset'
data_root_test = 'data/acdc/'

# test pipeline without test time augmentations
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# test pipeline with test time augmentations
img_ratios = [1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

val_dataloader = dict(
    batch_size=1, # This is overwritten in the final config
    num_workers=1, # This is overwritten in the final config
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type_test,
        data_root=data_root_test,
        data_prefix=dict(
            img_path='rgb_anon/test', seg_map_path='gt/test'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric',
    format_only=True,
    output_dir='work_dirs/acdc_testset_results')
test_evaluator = val_evaluator
