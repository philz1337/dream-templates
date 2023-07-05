import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image

pipe_txt2img = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe_txt2img.to("cuda")
image = pipe_txt2img("An image of a squirrel in Picasso style").images[0]
image.save("00_1.png")




# image = "image_of_squirrel_painting.png"
mask_url = "mask.png"
# image = Image.open(img_url)
mask_url = Image.open(mask_url)
image = image.resize((512, 512))
mask_image = mask_url.resize((512, 512))
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipe_inpaint = pipe_inpaint.to("cuda")
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe_inpaint(prompt=prompt, image=image, mask_image=mask_url).images[0]
image.save("00_2.png")




pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion", torch_dtype=torch.float16).to(
    "cuda"
)
prompt = "photo of a squirrel"
generator = torch.Generator(device="cuda").manual_seed(1024)
image_upscaled = pipe_img2img(prompt=prompt, image=image, strength=0.7, guidance_scale=7.5, generator=generator).images[0]
image_upscaled.save("00_3.png")