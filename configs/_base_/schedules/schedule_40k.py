iterations = 40000          # number of batches used in training
val_interval = 2000         # interval (number of iterations) for evaluation and checkpointing

# optimizer
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.norm': backbone_norm_multi,
    'backbone.pos_embed': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi,
}
custom_keys.update({
    f'backbone.blocks.{block_id}.norm1': backbone_norm_multi
    for block_id in range(24)
})
custom_keys.update({
    f'backbone.blocks.{block_id}.norm2': backbone_norm_multi
    for block_id in range(24)
})

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0)
)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=500,
        end=iterations,
        by_epoch=False)
]

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