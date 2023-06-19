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
)
from diffusers.utils import load_image

import settings
from stable_diffusion_controlnet_img2img import StableDiffusionControlNetImg2ImgPipeline
from controlnet_aux import MidasDetector, OpenposeDetector

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

        print("Loading pose...")
        self.openpose = OpenposeDetector.from_pretrained(
            "lllyasviel/ControlNet",
            cache_dir=settings.MODEL_CACHE,
        )

        print("Loading controlnet...")
        self.controlnet = ControlNetModel.from_pretrained(
            os.path.join(settings.MODEL_CACHE, "depth"),
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

        print("Loading controlnet openpose...")
        self.controlnet_openpose = ControlNetModel.from_pretrained(
            os.path.join(settings.MODEL_CACHE, "openpose"),
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")

        print("Loading controlnet tiles...")
        self.controlnet_tiles = ControlNetModel.from_pretrained(
            os.path.join(settings.MODEL_CACHE, "tiles"),
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")


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
        pipe = StableDiffusionPipeline.from_pretrained(
            weights_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker = None,
        ).to("cuda")
        
        start = time.time()
        pipe.load_textual_inversion("./ti/negative_hand-neg.pt", token="<negative-hand>")
        pipe.load_textual_inversion("./ti/badhandv4.pt", token="<badhandv4>")
        pipe.load_textual_inversion("./ti/easynegative.pt", token="<easynegative>")
        pipe.load_textual_inversion("./ti/ng_deepnegative_v1_75t.pt", token="<ng-deepnegative>")
        pipe.load_textual_inversion("./ti/bad-picture-chill-75v.pt", token="<bad-picture-chill>")
        pipe.load_textual_inversion("./ti/CyberRealistic_Negative-neg.pt.pt", token="<cyberrealistic-neg>")
        
        print("loading textual-inversions took: %0.2f" % (time.time() - start))

        return pipe

    def upscale(self, img, upscale_rate):
        w, h = img.size
        new_w, new_h = int(w * upscale_rate), int(h * upscale_rate)
        return img.resize((new_w, new_h), Image.BICUBIC)
    
    def resize_for_condition_image(self, input_image: Image, resolution: int):
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img

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

    def process_control_openpose(self, control_image_openpose):
        if control_image_openpose is None:
            return None

        return self.openpose(control_image_openpose)

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
                safety_checker=None,
                feature_extractor=pipe.feature_extractor,
            )

        if kind == "cnet_txt2img":
            return StableDiffusionControlNetPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=pipe.feature_extractor,
                controlnet=self.controlnet,
            )
        
        if kind == "cnet_txt2img_openpose":
            return StableDiffusionControlNetPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=pipe.feature_extractor,
                controlnet=self.controlnet_openpose,
            )

        if kind == "cnet_img2img":
            return StableDiffusionControlNetImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=pipe.feature_extractor,
                controlnet=self.controlnet,
            )
        
        if kind == "cnet_img2img_openpose":
            return StableDiffusionControlNetImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=pipe.feature_extractor,
                controlnet=self.controlnet_openpose,
            )
        
        if kind == "cnet_img2img_tiles":
            return StableDiffusionControlNetImg2ImgPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=pipe.feature_extractor,
                controlnet=self.controlnet_tiles,
            )

        if kind == "inpaint":
            return StableDiffusionInpaintPipelineLegacy(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                unet=pipe.unet,
                scheduler=pipe.scheduler,
                safety_checker=None,
                feature_extractor=pipe.feature_extractor,
            )

    @torch.inference_mode()
    def predict(
        self,
        control_image: Path = Input(
            description="Optional Image to use for guidance based on Midas depth",
            default=None,
        ),
        control_image_openpose: Path = Input(
            description="Optional Image to use for guidance based on Midas openpose",
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
                "DPM++ SDE Karras",
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
        upscale_rate: float = Input(
            description="Rate for Plain Upscaling. 1.0 corresponds to original image size", ge=1, le=20, default=1
        ),
        upscale_afterwards: bool = Input(
            description="upscale image after image generation", default=False
        ),
        upscale_afterwards_twice: bool = Input(
            description="upscale image after image generation twice, only works with tiles", default=False
        ),
        upscale_afterwards_rate: float = Input(
            description="Rate for Upscaling. 1.0 corresponds to original image size", ge=1, le=20, default=1
        ),
        upscale_afterwards_method: str = Input(
            default="img2img",
            choices=[
                "img2img",
                "tiles",
            ],
            description="Upscaler: Choose a scheduler."
        ),
        output_raw: bool = Input(
            description="Output the raw result when upscaling afterwards", default=False
        ),
        upscale_num_inference_steps: int = Input(
            description="Upscaler: Number of denoising steps", ge=1, le=500, default=20
        ),
        upscale_guidance_scale: float = Input(
            description="Upscaler: Scale for classifier-free guidance", ge=1, le=20, default=12
        ),
        upscale_prompt_strength: float = Input(
            description="Upscaler: Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.2,
        ),
        upscale_scheduler: str = Input(
            default="DDIM",
            choices=[
                "DDIM",
                "DPMSolverMultistep",
                "HeunDiscrete",
                "K_EULER_ANCESTRAL",
                "K_EULER",
                "KLMS",
                "PNDM",
                "UniPCMultistep",
                "DPM++ SDE Karras",
            ],
            description="Upscaler: Choose a scheduler."
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
        if control_image_openpose:
            control_image_openpose = self.load_image(control_image_openpose)
            control_image_openpose = self.process_control_openpose(control_image_openpose)
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
        if control_image and control_image_openpose:
            raise ValueError("Cannot use two different controlnets at the same time")
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
        elif control_image_openpose and image:
            print("Using ControlNet img2img with openpose")
            pipe = self.get_pipeline(pipe, "cnet_img2img_openpose")
            extra_kwargs = {
                "controlnet_conditioning_image": control_image_openpose,
                "image": image,
                "strength": prompt_strength,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds":negative_prompt_embeds
            }
        elif control_image_openpose:
            print("Using ControlNet txt2img with openpose")
            pipe = self.get_pipeline(pipe, "cnet_txt2img_openpose")
            extra_kwargs = {
                "image": control_image_openpose,
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

        if upscale_afterwards: 
            if upscale_afterwards_method == "img2img":
                print("Using upscale pipeline")
                upscale_pipe = self.get_pipeline(pipe, "img2img")
            elif upscale_afterwards_method == "tiles":
                print("Using upscale tiles pipeline")
                upscale_pipe = self.get_pipeline(pipe, "cnet_img2img_tiles")
            
        print("loading pipeline took: %0.2f" % (time.time() - start))

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        result_count = 0
        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(this_seed)
            output = pipe(
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                **extra_kwargs,
            )
            
            if upscale_afterwards:
                img = output.images[0]

                upscale_kwargs = {
                        "strength": upscale_prompt_strength,
                        "prompt_embeds": prompt_embeds,
                        "negative_prompt_embeds":negative_prompt_embeds
                    }
                
                if output_raw:
                    output_path = Path(f"/tmp/seed-{this_seed}-raw.png")
                    img.save(output_path)
                    yield Path(output_path)

                if upscale_afterwards_method == "img2img":
                    img = self.upscale(img, upscale_afterwards_rate)
                    upscale_pipe.scheduler = make_scheduler(upscale_scheduler, pipe.scheduler.config)
                    
                    output = upscale_pipe(
                        image=img,
                        guidance_scale=upscale_guidance_scale,
                        generator=generator,
                        num_inference_steps=upscale_num_inference_steps,
                        **upscale_kwargs,                
                    )

                elif upscale_afterwards_method == "tiles":
                    width_new_image = img.size[0]*upscale_afterwards_rate
                    condition_image = self.resize_for_condition_image(img, width_new_image)
                    output = upscale_pipe(                    
                        image=condition_image, 
                        controlnet_conditioning_image=condition_image, 
                        width=condition_image.size[0],
                        height=condition_image.size[1],
                        generator=generator,
                        num_inference_steps=upscale_num_inference_steps,
                        **upscale_kwargs,     
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
    scheduler_classes = {
        "DDIM": DDIMScheduler,
        "DPMSolverMultistep": DPMSolverMultistepScheduler,
        "HeunDiscrete": HeunDiscreteScheduler,
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
        "K_EULER": EulerDiscreteScheduler,
        "KLMS": LMSDiscreteScheduler,
        "PNDM": PNDMScheduler,
        "UniPCMultistep": UniPCMultistepScheduler,
        "DPM++ SDE Karras": DPMSolverMultistepScheduler
    }

    #default False
    karras_sigmas = False

    if name == "DPM++ SDE Karras":
        config.algorithm_type = 'sde-dpmsolver++'
        karras_sigmas = True
    elif name == "DPMSolverMultistep":
        config.algorithm_type = 'dpmsolver++'

    scheduler_class = scheduler_classes.get(name)
    if scheduler_class:
        print("Using Scheduler: {}".format(name), " with karras_sigmas: ", karras_sigmas)
        scheduler = scheduler_class.from_config(config, use_karras_sigmas=karras_sigmas)
        return scheduler
    else:
        raise ValueError("Invalid Scheduler Name: {}".format(name))
