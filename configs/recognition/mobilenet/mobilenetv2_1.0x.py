_base_ = [
    '../../_base_/models/mobilenet/mobilenetv2.py', '../../_base_/default_runtime.py'
]

#custom
pretrained = './pretrained/mobilenet3d/videoswin/kinetics_mobilenetv2_1.0x_RGB_16_best.pth'
model=dict(backbone=dict(width_mult=1., pretrained=pretrained),
           cls_head=dict(num_classes=4, loss_cls=dict(type='BCELoss')),
           neck=dict(loss_cls=dict(type='BCELoss')),
           test_cfg=dict(max_testing_views=4))

# dataset settings
dataset_type = 'VideoDataset'
# data_root = 'data/kinetics400/train'
# data_root_val = 'data/kinetics400/val'
# ann_file_train = 'data/kinetics400/kinetics400_train_list.txt'
# ann_file_val = 'data/kinetics400/kinetics400_val_list.txt'
# ann_file_test = 'data/kinetics400/kinetics400_val_list.txt'

#custom
ann_file_train = './datas/taillight/train.json'
ann_file_val = './datas/taillight/val.json'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
# train_pipeline = [
#     dict(type='DecordInit'),
#     dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
#     dict(type='DecordDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='RandomResizedCrop'),
#     dict(type='Resize', scale=(224, 224), keep_ratio=False),
#     dict(type='Flip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCTHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs', 'label'])
# ]

#custom
train_pipeline = [
    dict(type='FramesDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FlipCustom', flip_ratio=0.5),
    dict(type='GammaTrans', gamma_ratio=0.5),
    dict(type='Rotation', rotaion_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

#custom
val_pipeline = [
    dict(type='FramesDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FlipCustom', flip_ratio=0.5),
    dict(type='GammaTrans', gamma_ratio=0.5),
    dict(type='Rotation', rotaion_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

data = dict(
    videos_per_gpu=1,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=4
    ),
    # test_dataloader=dict(
    #     videos_per_gpu=1,
    #     workers_per_gpu=1
    # ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        # data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        # data_prefix=data_root_val,
        pipeline=val_pipeline),
    # test=dict(
    #     type=dataset_type,
    #     ann_file=ann_file_test,
    #     data_prefix=data_root_val,
    #     pipeline=test_pipeline)
)
# evaluation = dict(
#     interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

#custom
evaluation = dict(
    interval=1, metrics=['cls_accuracy', 'cls_precision_recall'])

# optimizer
# optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.),
#                                                  'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-5,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5
)

#uniformerv2 start
base_lr = 1e-4
optimizer = dict(type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01,
                paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.0),
                                                'bias': dict(decay_mult=0.0),}))

# optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))


#uniformerv2 end

total_epochs = 25

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/mobilenetv2_1.0x.py'
find_unused_parameters = True

