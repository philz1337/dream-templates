import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline

print("torch version:   " , torch.__version__)

img_path = "x.png"
mask_path = "mask.png"

init_image = PIL.Image.open(img_path).convert("RGB")
mask_image = PIL.Image.open(mask_path).convert("RGB")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "RAW photo, sks man, (high detailed skin:1.2), as a business man, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image,
             width=512, height=768, negative_prompt=negative_prompt, guidance_scale=4
             ).images[0]

output_path = "output.png"
image.save(output_path)
