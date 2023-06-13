from diffusers import StableDiffusionPipeline
import torch

model_id = "philz1337/rv2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16).to("cuda")


pipe.load_textual_inversion("./ti/negative_hand-neg.pt", token="<ti-neghand>")
pipe.load_textual_inversion("./ti/badhandv4.pt", token="<badhandv4>")
pipe.load_textual_inversion("./ti/easynegative.pt", token="<easynegative>")
pipe.load_textual_inversion("./ti/ng_deepnegative_v1_75t.pt", token="<ng_deepnegative>")
pipe.load_textual_inversion("./ti/pureerosface_v1.pt", token="<pureerosface>")

pipe.load_textual_inversion("./ti/angry512.pt", token="<em-angry>")
pipe.load_textual_inversion("./ti/happy512.pt", token="<em-happy>")
pipe.load_textual_inversion("./ti/shock512.pt", token="<em-shock>")
pipe.load_textual_inversion("./ti/smile512.pt", token="<em-smile>")

prompt = "woman with beautiful face, close up, beauty marks, sandy skin, photo, detailed,(moody lighting), (sharp focus), "
negative_prompt= "<ti-neghand>"

generator = torch.Generator(device="cuda").manual_seed(3336)
image = pipe(prompt, generator=generator, negative_prompt=negative_prompt, num_inference_steps=50).images[0]
image.save("cat-backpack.png")