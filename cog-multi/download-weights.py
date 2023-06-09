#!/usr/bin/env python

import os
import shutil

import settings

if os.path.exists(settings.MODEL_CACHE):
    shutil.rmtree(settings.MODEL_CACHE)
os.makedirs(settings.MODEL_CACHE)

import torch
from diffusers import ControlNetModel
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


shutil.rmtree(TMP_CACHE)


# Textual inversion
from huggingface_hub import hf_hub_download

with open("textual-inversion-concepts.txt") as infile:
    CONCEPTS = [line.rstrip() for line in infile]

print("Downloading pre-trained concepts...")
for concept in CONCEPTS:
    concept = concept.split(":")[0]
    os.makedirs(concept, exist_ok=True)
    embeds_path = hf_hub_download(repo_id=concept, filename="learned_embeds.bin", cache_dir=concept)
    token_path = hf_hub_download(repo_id=concept, filename="token_identifier.txt", cache_dir=concept)

    with open(token_path, 'r') as file:
        placeholder = file.read()
    print(f"{concept}: {placeholder}")