from diffusers import StableDiffusionPipeline
import torch

model_id = "SG161222/Realistic_Vision_V2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

pipe.load_textual_inversion("negative_hand-neg.pt")

prompt = "A wonderfull woman"
negative_prompt = "negative_hand-neg"

image = pipe(prompt, negative_prompt, num_inference_steps=50).images[0]
image.save("cat-backpack.png")