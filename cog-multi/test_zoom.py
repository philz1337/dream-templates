import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline

device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion", torch_dtype=torch.float16).to(
    device
)
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

pipe.to("cuda")

image = pipe("An image of a squirrel in Picasso style").images[0]

image.save("image_of_squirrel_painting.png")

prompt = "photo of a squirrel"
generator = torch.Generator(device=device).manual_seed(1024)
image_upscaled = pipe(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5, generator=generator).images[0]
image_upscaled.save("upscaled.png")