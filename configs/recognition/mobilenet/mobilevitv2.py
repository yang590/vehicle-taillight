_base_ = [
    '../../_base_/models/mobilenet/mobilevitv2.py', '../../_base_/default_runtime.py'
]

#custom
pretrained = './pretrained/mobilevitv2/im21k_ft1k/mobilevitv2-1.5.pt'
# pretrained = './pretrained/mobilevitv2/im21k_ft1k/mobilevitv2-2.0.pt'
# pretrained = './pretrained/mobilevitv2/im1k/train256x256_ft384x384/mobilevitv2-1.0.pt'
# pretrained = './pretrained/mobilevitv2/im1k/train256x256_ft384x384/mobilevitv2-0.5.pt'
model=dict(backbone=dict(pretrained=pretrained, width_multiplier=1.5),
           cls_head=dict(in_channels=768, num_classes=4, loss_cls=dict(type='BCELoss')),
           # neck=dict(in_channels=[576], out_channels=4, loss_weight=0.5, loss_cls=dict(type='BCELoss')),
           neck=None,
           test_cfg=dict(max_testing_views=4, average_clips='score'))

dataset_type = 'VideoDataset'

#custom
data_root = './carlight/train/all/'
ann_file_train = './carlight/train/train.json'
ann_file_val = './carlight/train/val.json'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


#custom
train_pipeline = [
    dict(type='FramesDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='FlipCustom', flip_ratio=0.5),
    # dict(type='GammaTrans', gamma_ratio=0.5),
    # dict(type='Rotation', rotaion_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

#custom
val_pipeline = [
    dict(type='FramesDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # dict(type='FlipCustom', flip_ratio=0.5),
    # dict(type='GammaTrans', gamma_ratio=0.5),
    # dict(type='Rotation', rotaion_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=4
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=val_pipeline),
)
# evaluation = dict(
#     interval=1, metrics=['top_k_accuracy', 'cls_precision_recall'])

#custom
evaluation = dict(
    interval=1, metrics=['mean_cls_accuracy', 'aprf_custom'])


base_lr = 1e-4
optimizer = dict(type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.02,
                paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.0),
                                                'bias': dict(decay_mult=0.0),}))

# optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)


total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/mobilevitv2_2.0x.py'
find_unused_parameters = True

