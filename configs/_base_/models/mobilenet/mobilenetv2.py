# model settings
model = dict(
    type='MobileRec3D',
    backbone=dict(
        type='MobileNetV2Custom',
        sample_size=224,
        width_mult=1.),
    cls_head=dict(
        type='MobileHead',
        in_channels=1280,
        num_classes=4,
        loss_cls=dict(type='CrossEntropyLoss'),
        dropout_ratio=0.5,
        init_std=0.01),
    neck=dict(
        type='MobileNeck',
        in_channels=[32, 96, 320],
        out_channels=4,
        loss_weight=0.5,
        loss_cls=dict(type='CrossEntropyLoss')),
    test_cfg = dict(average_clips='score'))