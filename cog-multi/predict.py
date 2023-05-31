import os
import shutil
import subprocess
from typing import Iterator
import time


import torch
from cog import BasePredictor, Input, Path
import io
from compel import Compel
import requests
import tarfile
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
    StableDiffusionLatentUpscalePipeline
)
from diffusers.utils import load_image

import settings
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from controlnet_aux import MidasDetector

from PIL import Image
import numpy as np
from functools import lru_cache
from weights import WeightsDownloadCache

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading midas...")
        self.midas = MidasDetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir=settings.MODEL_CACHE
        )

        print("Loading controlnet...")
        self.controlnet = ControlNetModel.from_pretrained(
            os.path.join(settings.MODEL_CACHE, "depth"),
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

        start = time.time()
        print("Loading latent upscaler...")
        self.latent_upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", 
            torch_dtype=torch.float16,
            local_files_only=False,
        ).to("cuda")
        print("loading latent upscaler took: %0.2f" % (time.time() - start))


        self.weights_download_cache = WeightsDownloadCache()

    def get_weights(self, weights: str):
        if weights.startswith("https://"):
            url = weights
        else:
            url = f"https://storage.googleapis.com/replicant-misc/{weights}.tar"

        if 'replicate.delivery' in url:
            url = url.replace('replicate.delivery/pbxt', 'storage.googleapis.com/replicate-files')

        path = self.weights_download_cache.ensure(url)
        return self.gpu_weights(path)

    @lru_cache(maxsize=10)
    def gpu_weights(self, weights_path: str):
        print(f"Loading txt2img... {weights_path}")
        return StableDiffusionPipeline.from_pretrained(
            weights_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

    def upscale(self, img, upscale_rate):
        w, h = img.size
        new_w, new_h = int(w * upscale_rate), int(h * upscale_rate)
        return img.resize((new_w, new_h), Image.BICUBIC)
    
    def load_image(self, image_path: Path, upscale_rate: float = 1.0):
        if image_path is None:
            return None
        # not sure why I have to copy the image, but it fails otherwise
        # seems like a bug in cog
        if os.path.exists("img.png"):
            os.unlink("img.png")
        shutil.copy(image_path, "img.png")
                
        if upscale_rate > 1.0:
            # Call the upscale function on the loaded image
            img = Image.open("img.png")
            img = self.upscale(img, upscale_rate)
        else:
            # Load the image directly without upscaling
            img = Image.open("img.png")
        
        return img

    def process_control(self, control_image):
        if control_image is None:
            return None

        depth_image, normal_image = self.midas(control_image)

        # https://github.com/patrickvonplaten/controlnet_aux/issues/7
        image = np.array(depth_image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def get_pipeline(self, pipe, kind):
        if kind == "txt2img":
            return pipe

        if kind == "img2img":
            return StableDiffusionImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
            )

        if kind == "cnet_txt2img":
            return StableDiffusionControlNetPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
                controlnet=self.controlnet,
            )

        if kind == "cnet_img2img":
            return StableDiffusionControlNetImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
                controlnet=self.controlnet,
            )

        if kind == "inpaint":
            return StableDiffusionInpaintPipelineLegacy(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=pipe.safety_checker,
                feature_extractor=pipe.feature_extractor,
            )
        # if kind == "latent_upscale":
        #     return StableDiffusionLatentUpscalePipeline(
        #         os.path.join(settings.MODEL_CACHE, "models--stabilityai--sd-x2-latent-upscaler"),
        #         # vae=pipe.vae,
        #         # text_encoder=pipe.text_encoder,
        #         # tokenizer=pipe.tokenizer,
        #         # unet=pipe.unet,
        #         # scheduler=pipe.scheduler,
        #         safety_checker=pipe.safety_checker,
        #         # feature_extractor=pipe.feature_extractor,
        #     )

    @torch.inference_mode()
    def predict(
        self,
        control_image: Path = Input(
            description="Optional Image to use for guidance based on Midas depth",
            default=None,
        ),
        weights: str = Input(
            description="Which weights to use",
        ),
        image: Path = Input(
            description="Optional Image to use for img2img guidance", default=None
        ),
        mask: Path = Input(
            description="Optional Mask to use for legacy inpainting", default=None
        ),
        prompt: str = Input(
            description="Input prompt",
            default="photo of cjw person",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=32,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "DPMSolverMultistep",
                "HeunDiscrete",
                "K_EULER_ANCESTRAL",
                "K_EULER",
                "KLMS",
                "PNDM",
                "UniPCMultistep",
            ],
            description="Choose a scheduler.",
        ),
        disable_safety_check: bool = Input(
            description="Disable safety check. Use at your own risk!", default=False
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        info: bool = Input(
            description="log extra information about the run", default=False
        ),
        upscaler: str = Input(
            default="Plain",
            choices=[
                "Plain",
            ],
            description="Choose a upscaler.",
        ),
        upscale_rate: float = Input(
            description="Rate for Upscaling. 1.0 corresponds to original image size", ge=1, le=20, default=1
        ),
        latent_upscale: int = Input(
            description="Rate for Latent Upscaling. 1.0 corresponds to original image size.",
            choices=[1, 2, 4],
            default=1,
        ),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""

        if info:
            print('GPU GPU GPU')
            print(self.gpu_weights.cache_info())
            os.system("nvidia-smi")

            print('DISK DISK DISK')
            os.system("df -h")
            os.system("free -h")
            print(self.weights_download_cache.cache_info())

        start = time.time()
        pipe = self.get_weights(weights)
        print("loading weights took: %0.2f" % (time.time() - start))


        start = time.time()
        if image:
             image = self.load_image(image,upscale_rate)
        if control_image:
            control_image = self.load_image(control_image)
            control_image = self.process_control(control_image)
        if mask:
            mask = self.load_image(mask)
        print("loading images took: %0.2f" % (time.time() - start))

        # FIXME(ja): we shouldn't need to do this multiple times
        # or perhaps we create the object each time?
        

        print("Loading compel...")
        compel = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
        )

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

        start = time.time()
        if control_image and mask:
            raise ValueError("Cannot use controlnet and inpainting at the same time")
        elif control_image and image:
            print("Using ControlNet img2img")
            pipe = self.get_pipeline(pipe, "cnet_img2img")
            extra_kwargs = {
                "controlnet_conditioning_image": control_image,
                "image": image,
                "strength": prompt_strength,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds":negative_prompt_embeds
            }
        elif control_image:
            print("Using ControlNet txt2img")
            pipe = self.get_pipeline(pipe, "cnet_txt2img")
            extra_kwargs = {
                "image": control_image,
                "width": width,
                "height": height,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds":negative_prompt_embeds
            }
        elif image and mask:
            print("Using inpaint pipeline")
            pipe = self.get_pipeline(pipe, "inpaint")
            # FIXME(ja): prompt/negative_prompt are sent to the inpainting pipeline
            # because it doesn't support prompt_embeds/negative_prompt_embeds
            extra_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "mask_image": mask,
                "strength": prompt_strength,
            }
        elif image:
            print("Using img2img pipeline")
            pipe = self.get_pipeline(pipe, "img2img")
            extra_kwargs = {
                "image": image,
                "strength": prompt_strength,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds":negative_prompt_embeds
            }
        else:
            print("Using txt2img pipeline")
            pipe = self.get_pipeline(pipe, "txt2img")
            extra_kwargs = {
                "width": width,
                "height": height,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds":negative_prompt_embeds
            }

        print("loading pipeline took: %0.2f" % (time.time() - start))

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        if disable_safety_check:
            pipe.safety_checker = None

        result_count = 0
        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(this_seed)

            if latent_upscale == 2:
                low_res_latents = pipe(
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    output_type="latent",
                    **extra_kwargs,
                ).images

                output = self.latent_upscaler(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=low_res_latents,
                    num_inference_steps=20,
                    guidance_scale=0,
                    generator=generator,
                )

            elif latent_upscale == 4:
                print("upscaling x4 not working yet")
                low_res_latents = pipe(
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    output_type="latent",
                    **extra_kwargs,
                ).images
                
                output = self.latent_upscaler(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=low_res_latents,
                    num_inference_steps=20,
                    guidance_scale=0,
                    generator=generator,
                )
                
            else:
                output = pipe(
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    **extra_kwargs,
                )

            



            if output.nsfw_content_detected and output.nsfw_content_detected[0]:
                continue

            output_path = f"/tmp/seed-{this_seed}.png"
            output.images[0].save(output_path)
            yield Path(output_path)
            result_count += 1

        if result_count == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )


def make_scheduler(name, config):
    return {
        "DDIM": DDIMScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        "HeunDiscrete": HeunDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "PNDM": PNDMScheduler.from_config(config),
        "UniPCMultistep": UniPCMultistepScheduler.from_config(config),
    }[name]
