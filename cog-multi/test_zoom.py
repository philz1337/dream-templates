import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from compel import Compel

def upscale(img, upscale_rate):
    w, h = img.size
    new_w, new_h = int(w * upscale_rate), int(h * upscale_rate)
    return img.resize((new_w, new_h), Image.BICUBIC)



image = "image_of_squirrel_painting.png"
mask_url = "mask.png"
image = Image.open(image)
mask_url = Image.open(mask_url)
image = image.resize((512, 512))
mask_image = mask_url.resize((512, 512))
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe_inpaint = pipe_inpaint.to("cuda")
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe_inpaint(prompt=prompt, image=image, mask_image=mask_url, num_inference_steps=15).images[0]
image.save("00_2.png")

upscale_image= upscale(image, 1.5)





compel = Compel(
            tokenizer=pipe_inpaint.tokenizer,
            text_encoder=pipe_inpaint.text_encoder,
        )

prompt = "photo of a squirrel"
negative_prompt = "anime"


if prompt:
            print("parsed prompt:", compel.parse_prompt_string(prompt))
            prompt_embeds = compel(prompt)
else:
            prompt_embeds = None

if negative_prompt:
            print(
                "parsed negative prompt:",
                compel.parse_prompt_string(negative_prompt),
            )
            negative_prompt_embeds = compel(negative_prompt)
else:
            negative_prompt_embeds = None
            
            

upscale_kwargs = {
                        "prompt_embeds": prompt_embeds,
                        "negative_prompt_embeds":negative_prompt_embeds
                    }

pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion", torch_dtype=torch.float16).to(
    "cuda"
)

generator = torch.Generator(device="cuda").manual_seed(1024)
image_upscaled = pipe_img2img(image=upscale_image, strength=0.5, guidance_scale=7.5,num_inference_steps=10,  generator=generator, **upscale_kwargs).images[0]
image_upscaled.save("00_3.png")