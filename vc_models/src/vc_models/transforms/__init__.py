#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.transforms as T

from vc_models.transforms.to_tensor_if_not import ToTensorIfNot
from vc_models.transforms.random_shifts_aug import RandomShiftsAug
from vc_models.transforms.randomize_env_transform import RandomizeEnvTransform


def vit_transforms(resize_size=256, output_size=224):
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(output_size),
            ToTensorIfNot(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

# https://github.com/huggingface/transformers/blob/307c5238546ba1675daabc46050c63ffde25f8e6/src/transformers/models/dinov3_vit/image_processing_dinov3_vit_fast.py#L74
def dinov3_vit_transforms(output_size=224):
    return T.Compose(
        [
            T.Resize(output_size, interpolation=T.InterpolationMode.BILINEAR),
            ToTensorIfNot(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

def resnet_transforms(resize_size=256, output_size=224):
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(output_size),
            ToTensorIfNot(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# based on https://github.com/facebookresearch/r3m/blob/main/r3m/example.py#L24
def r3m_transforms(resize_size=256, output_size=224):
    return T.Compose(
        [
            T.Resize(resize_size), # interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(output_size),
            T.ToTensor(),                # converts to [0,1]
            T.Lambda(lambda x: x * 255), # scale back to [0,255]
        ]
    )


def clip_transforms(resize_size=256, output_size=224):
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(output_size),
            ToTensorIfNot(),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

def deit_transforms(output_size=224):
    """
    TheiaModel, backbone is Deit
    model.backbone.processor
ViTImageProcessor {
  "do_convert_rgb": null,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "ViTImageProcessor",
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 224,
    "width": 224
  }
}
    """
    return T.Compose([
        # Resize directly to (224, 224) with bilinear interpolation
        T.Resize((output_size, output_size), interpolation=T.InterpolationMode.BILINEAR),

        # Convert to tensor and scale [0,255] → [0,1]
        ToTensorIfNot(),

        # Normalize to [-1,1] using mean=std=0.5
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def transform_augment(
    # Resize/crop
    resize_size=256,
    output_size=224,
    # Jitter
    jitter=True,
    jitter_prob=1.0,
    jitter_brightness=0.3,
    jitter_contrast=0.3,
    jitter_saturation=0.3,
    jitter_hue=0.3,
    # Shift
    shift=True,
    shift_pad=4,
    # Randomize environments
    randomize_environments=False,
    normalize=False,
):
    transforms = [ToTensorIfNot(), T.Resize(resize_size), T.CenterCrop(output_size)]

    if jitter:
        transforms.append(
            T.RandomApply(
                [
                    T.ColorJitter(
                        jitter_brightness,
                        jitter_contrast,
                        jitter_saturation,
                        jitter_hue,
                    )
                ],
                p=jitter_prob,
            )
        )

    if shift:
        transforms.append(RandomShiftsAug(shift_pad))
    
    if normalize:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    transforms = T.Compose(transforms)

    return RandomizeEnvTransform(
        transforms, randomize_environments=randomize_environments
    )