#!/usr/bin/env python

import os
import shutil

import settings

if os.path.exists(settings.MODEL_CACHE):
    shutil.rmtree(settings.MODEL_CACHE)
os.makedirs(settings.MODEL_CACHE)

import torch
from diffusers import ControlNetModel, StableDiffusionInpaintPipeline
from controlnet_aux import MidasDetector

MidasDetector.from_pretrained(
    "lllyasviel/ControlNet",
    cache_dir=settings.MODEL_CACHE,
)

TMP_CACHE = "tmp_cache"

if os.path.exists(TMP_CACHE):
    shutil.rmtree(TMP_CACHE)
os.makedirs(TMP_CACHE)


cn = ControlNetModel.from_pretrained(
    settings.CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    cache_dir=TMP_CACHE,
)
cn.half()
cn.save_pretrained(os.path.join(settings.MODEL_CACHE, 'depth'))

cn_pose = ControlNetModel.from_pretrained(
    settings.CONTROLNET_MODEL_OPENPOSE,
    torch_dtype=torch.float16,
    cache_dir=TMP_CACHE,
)
cn_pose.half()
cn_pose.save_pretrained(os.path.join(settings.MODEL_CACHE, 'openpose'))

cn_tiles = ControlNetModel.from_pretrained(
    settings.CONTROLNET_MODEL_TILES,
    torch_dtype=torch.float16,
    cache_dir=TMP_CACHE,
)
cn_tiles.half()
cn_tiles.save_pretrained(os.path.join(settings.MODEL_CACHE, 'tiles'))

inpainting = StableDiffusionInpaintPipeline.from_pretrained(
    settings.INPAINTING_MODEL,
    torch_dtype=torch.float16,
    cache_dir=TMP_CACHE,
)
inpainting.save_pretrained(os.path.join(settings.MODEL_CACHE, 'inpainting'))


shutil.rmtree(TMP_CACHE)