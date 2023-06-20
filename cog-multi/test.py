from diffusers import DiffusionPipeline
import torch
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler

guided_pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    custom_pipeline="stable_diffusion_reference",
    safety_checker=None,
    torch_dtype=torch.float16
)
guided_pipeline.enable_attention_slicing()
guided_pipeline = guided_pipeline.to("cuda")

guided_pipeline.scheduler = UniPCMultistepScheduler.from_config(guided_pipeline.scheduler.config)


input_image = load_image("https://janaprokhorenko.de/wp-content/uploads/2022/11/Tochter-und-Vater-768x1186.png")


result_img = guided_pipeline(ref_image=input_image,
      prompt="oilpainting of beautiful girl",
      width=512,
      height=768,
      num_inference_steps=20,
      reference_attn=True,
      reference_adain=True).images[0]

result_img.save("result.png")