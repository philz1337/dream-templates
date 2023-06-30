import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import requests

url = "https://civitai.com/api/download/models/15603"
filename = "light_and_shadow.safetensors"

response = requests.get(url)
response.raise_for_status()

with open(filename, "wb") as file:
    file.write(response.content)

print("Download abgeschlossen!")


pipeline = StableDiffusionPipeline.from_pretrained(
    "gsdf/Counterfeit-V2.5", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config, use_karras_sigmas=True
)

#pipeline.load_lora_weights(".", weight_name="light_and_shadow.safetensors")