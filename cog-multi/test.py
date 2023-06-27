import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline

print("torch version:   " , torch.__version__)

img_path = "image.jpeg"
mask_path = "mask.png"

init_image = PIL.Image.open(img_path).convert("RGB")
mask_image = PIL.Image.open(mask_path).convert("RGB")

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image,
             width=512, height=768
             ).images[0]

output_path = "output.png"
image.save(output_path)
