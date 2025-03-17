iterations = 1000           # number of batches used in training
val_interval = 500          # interval (number of iterations) for evaluation and checkpointing

# optimizer
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.0, decay_mult=0.0),
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi,
}

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0)
)

# training schedule for 5k iterations
train_cfg = dict(type='IterBasedTrainLoop', max_iters=iterations, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval, save_last=True, published_keys=['meta', 'state_dict'], max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)