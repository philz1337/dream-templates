import torch
from diffusers import (UniPCMultistepScheduler, DiffusionPipeline)
from diffusers.utils import load_image

input_image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")

pipe = DiffusionPipeline.from_pretrained(
       repo_name="runwayml/stable-diffusion-v1-5",
       custom_pipeline="stable_diffusion_reference.py",
       safety_checker=None,
       torch_dtype=torch.float16
       ).to('cuda:0')

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

result_img = pipe(ref_image=input_image,
      prompt="1girl",
      num_inference_steps=20,
      reference_attn=True,
      reference_adain=True).images[0]