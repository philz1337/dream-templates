from diffusers import StableDiffusionPipeline
import torch

model_id = "philz1337/rv2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, cache_dir="ph-temp", safety_checker=None, torch_dtype=torch.float16).to("cuda")

#pipe.load_textual_inversion("./awaitingtongue.pt", token="<ti-awaitingtongue>",  local_files_only=True, )
#pipe.load_textual_inversion("./bukkakAI.pt", token="<ti-bukkak>")
pipe.load_textual_inversion("./negative_hand-neg.pt", token="<ti-neghand>")
pipe.load_textual_inversion("./EmWat69.pt", token="<ti-em>")

prompt = "<ti-neghand>, hands, showing hands, beauty marks, sandy skin, photo, detailed hands,(moody lighting), (sharp focus), "
negative_prompt= "<ti-neghand>"

generator = torch.Generator(device="cuda").manual_seed(1337)

image = pipe(prompt, generator=generator, negative_prompt=negative_prompt, num_inference_steps=30).images[0]
image.save("cat-backpack.png")