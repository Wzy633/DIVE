# dataset settings
dataset_type = 'CocoDataset'

backend_args = None
# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities'))
]


#================DOTAv2==================
data_root = '../data/LAE-FOD/DOTAv2/'
metainfo = dict(
    classes = ('plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool', 'container crane', 'airport', 'helipad')
)
DOTAv2_train_dataset=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='processed_DOTAv2_train.json',
    metainfo=metainfo,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=train_pipeline,
    return_classes=True,
    backend_args=backend_args)
DOTAv2_val_dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='DOTAv2_val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        metainfo=metainfo,
        pipeline=test_pipeline,
        return_classes=True,
        backend_args=backend_args)
DOTAv2_val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'DOTAv2_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

dataset_prefixes = ['DOTAv2']

all_train_dataset = [DOTAv2_train_dataset]

all_val_dataset = [DOTAv2_val_dataset]
all_metrics = [DOTAv2_val_evaluator]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=all_train_dataset))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='ConcatDataset', datasets=all_val_dataset))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='MultiDatasetsEvaluator',
    metrics=all_metrics,
    dataset_prefixes=dataset_prefixes)
test_evaluator = val_evaluator