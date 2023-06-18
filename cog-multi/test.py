
import torch
from PIL import Image
from diffusers import ControlNetModel, DiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    H, W = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS if k > 1 else Image.AREA)
    return img

controlnet = ControlNetModel.from_pretrained('takuma104/control_v11', 
                                             subfolder='control_v11f1e_sd15_tile',
                                             torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    custom_pipeline="stable_diffusion_controlnet_img2img",
    controlnet=controlnet,
    torch_dtype=torch.float16).to('cuda')
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

source_image = load_image('https://github.com/lllyasviel/ControlNet-v1-1-nightly/raw/main/test_imgs/dog64.png')

condition_image = resize_for_condition_image(source_image, 1024)
image = pipe(prompt="best quality", 
     negative_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality", 
     image=condition_image, 
     controlnet_conditioning_image=condition_image, 
     width=condition_image.size[0],
     height=condition_image.size[1],
     strength=1.0,
     generator=torch.manual_seed(0),
     num_inference_steps=32,
     ).images[0]
image.save("tiles.png")