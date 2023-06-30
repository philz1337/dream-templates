import torch

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import requests
import os

url = "https://civitai.com/api/download/models/15603"
filename = "light_and_shadow.safetensors"
folder_path = "lora"

# Erstelle den Ordner, falls er noch nicht existiert
os.makedirs(folder_path, exist_ok=True)

file_path = os.path.join(folder_path, filename)

response = requests.get(url)
response.raise_for_status()  # Überprüft, ob der Download erfolgreich war

with open(file_path, "wb") as file:
    file.write(response.content)
print("Download abgeschlossen! Datei gespeichert unter:", file_path)

pipeline = StableDiffusionPipeline.from_pretrained(
    "gsdf/Counterfeit-V2.5", torch_dtype=torch.float16, safety_checker=None, seed=42
).to("cuda")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config, use_karras_sigmas=True
)

#pipeline.load_lora_weights(".", weight_name="light_and_shadow.safetensors")