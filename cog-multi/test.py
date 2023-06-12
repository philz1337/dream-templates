from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

pipe.load_textual_inversion("./awaitingtongue.pt")

prompt = "A <cat-toy> backpack"

image = pipe(prompt, num_inference_steps=30).images[0]
image.save("cat-backpack.png")