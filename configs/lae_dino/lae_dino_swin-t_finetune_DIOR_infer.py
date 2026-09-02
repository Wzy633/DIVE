"""Inference-friendly config wrapper for DetInferencer.

The training config `lae_dino_swin-t_finetune_DIOR.py` uses `ConcatDataset`
for `test_dataloader`, which does not expose `.dataset.pipeline` at the top
level. `DetInferencer` expects `cfg.test_dataloader.dataset.pipeline`.

This wrapper keeps the same model settings but provides a minimal, explicit
`test_dataloader.dataset.pipeline` for inference usage.
"""

_base_ = ['./lae_dino_swin-t_finetune_DIOR.py']

backend_args = None
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'text', 'custom_entities'),
    ),
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

