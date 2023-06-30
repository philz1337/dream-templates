import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import requests
import os
from urllib.parse import urlparse

def download_lora_weights(url: str):
        folder_path = "lora"

        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        if "civitai.com" in parsed_url.netloc:
            filename = f"{os.path.basename(parsed_url.path)}.safetensors"

        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, filename)

        response = requests.get(url)
        response.raise_for_status()

        with open(file_path, "wb") as file:
            file.write(response.content)

        print("Lora saved under:", file_path)
        return file_path

url = "https://civitai.com/api/download/models/7870"

lora_file_path = download_lora_weights(url)

pipeline = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V3.0_VAE", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config, use_karras_sigmas=True
)

pipeline.load_lora_weights(".", weight_name=lora_file_path)

prompt = "an actress (standing behind a podium), shirtlifting, accepting an (award), holding a trophy"
image = pipeline(
    prompt, 
    generator= torch.Generator("cuda").manual_seed(3),
    cross_attention_kwargs={"scale": 1}
                ).images[0]
image.save("000000_result_test.png")