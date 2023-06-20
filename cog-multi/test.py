from diffusers import DiffusionPipeline
import torch
from diffusers.utils import load_image

guided_pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    custom_pipeline="stable_diffusion_reference",
    safety_checker=None,
    torch_dtype=torch.float16
)
guided_pipeline.enable_attention_slicing()
guided_pipeline = guided_pipeline.to("cuda")

input_image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")


result_img = guided_pipeline(ref_image=input_image,
      prompt="1girl",
      num_inference_steps=20,
      reference_attn=True,
      reference_adain=True).images[0]