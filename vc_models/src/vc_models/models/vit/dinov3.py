# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
import os
import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import torch
import torch.nn.init
from torch import Tensor, nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr
import sys
sys.path.append('/home/dongwang/workspace/github/clip/dinov3/')
from dinov3.models.vision_transformer import DinoVisionTransformer
import types
import numpy as np

def dinov3_forward_features(self, x):
    x = self.original_forward_features(x)
    if self.global_pool:
        outcome = x["x_norm_patchtokens"]
        outcome = outcome.mean(dim=1)  # global pool without cls token
        outcome = self.norm(x)
    elif self.use_cls:
        if self.avg_cls_reg:
            outcome = torch.cat((x["x_norm_clstoken"].unsqueeze(1), x["x_storage_tokens"]), dim=1).to(x["x_storage_tokens"].device).mean(dim=1)
            outcome = self.norm(x)
        else:
            outcome = x["x_norm_clstoken"]
    elif self.flatten_embedding:
        outcome = x["x_norm_patchtokens"]
        outcome = outcome.reshape(outcome.shape[0], -1)
    else:
        from vc_models.models.vit import reshape_embedding
        outcome = x["x_norm_patchtokens"]
        outcome = reshape_embedding(outcome)
    return outcome


def vit_large_patch16(global_pool=False, use_cls=True, reg_tokens=0, flatten_embedding=False,
        avg_cls_reg=False, **kwargs):
    embed_dim = kwargs.pop("embed_dim", 1024)
    model = DinoVisionTransformer(
        patch_size=16,
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        qkv_bias=True,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.global_pool = global_pool
    model.use_cls = use_cls
    model.reg_tokens = reg_tokens
    model.flatten_embedding = flatten_embedding
    model.avg_cls_reg = avg_cls_reg
    model.embed_dim = embed_dim

    if global_pool:
        model.classifier_feature = "global_pool"
    elif use_cls:
        model.classifier_feature = "use_cls_token"
    else:
        model.classifier_feature = "reshape_embedding"
    return model

def vit_huge_patch16(global_pool=False, use_cls=True, reg_tokens=0, flatten_embedding=False,
        avg_cls_reg=False, **kwargs):
    embed_dim = kwargs.pop("embed_dim", 1280)
    model = DinoVisionTransformer(
        patch_size=16,
        embed_dim=embed_dim,
        depth=32,
        num_heads=20,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    model.global_pool = global_pool
    model.use_cls = use_cls
    model.reg_tokens = reg_tokens
    model.flatten_embedding = flatten_embedding
    model.avg_cls_reg = avg_cls_reg
    model.embed_dim = embed_dim

    if global_pool:
        model.classifier_feature = "global_pool"
    elif use_cls:
        model.classifier_feature = "use_cls_token"
    else:
        model.classifier_feature = "reshape_embedding"
    return model

def load_mae_encoder(model, checkpoint_path=None, subkey="model", **kwargs):

    if not os.path.isabs(checkpoint_path):
        model_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','..')
        checkpoint_path = os.path.join(model_base_dir,checkpoint_path)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model.original_forward_features = model.forward_features
    model.forward_features = types.MethodType(dinov3_forward_features, model)
    model.forward = types.MethodType(dinov3_forward_features, model)

    if not model.global_pool and not model.use_cls:
        model.final_spatial = int(model.patch_embed.num_patches**0.5)
        model.embed_dim = (model.final_spatial, model.final_spatial, model.embed_dim)
        if model.flatten_embedding:
            model.embed_dim = np.prod(model.embed_dim)
    return model
