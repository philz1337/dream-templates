from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

pipe.to("cuda")

image = pipe("An image of a squirrel in Picasso style").images[0]

image.save("image_of_squirrel_painting.png")