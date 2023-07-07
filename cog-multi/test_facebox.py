from PIL import Image
import face_recognition
import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
import cv2

from PIL import Image
import face_recognition

def find_face(image_link):    
    image = face_recognition.load_image_file(image_link)

    face_locations = face_recognition.face_locations(image)

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    if face_locations:
        face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[3] - loc[1]))

        top, right, bottom, left = face_location
        print("The largest is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # Berechnung der Größe des ausgeschnittenen Gesichtsbereichs
        width = right - left
        height = bottom - top

        # Vergrößern des Gesichtsbereichs um 20%
        width_increase = int(width * 0.2)
        height_increase = int(height * 0.2)

        # Anpassen der Koordinaten um den zusätzlichen Bereich
        top -= height_increase
        right += width_increase
        bottom += height_increase
        left -= width_increase

        # Überprüfung, ob die Koordinaten innerhalb der Bildgrenzen liegen
        top = max(0, top)
        right = min(image.shape[1], right)
        bottom = min(image.shape[0], bottom)
        left = max(0, left)

        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)

    else:
        print("No faces found in the photograph.")
        return None, None, None

    return pil_image, top, left

def preScale(img):
    w, h = img.size
    if w > h:
        new_w = 512
        new_h = int(h * (new_w / w))
    else:
        new_h = 512
        new_w = int(w * (new_h / h))
    return img.resize((new_w, new_h), Image.BICUBIC)

def faceImg2img(rescaled_face):
    prompt = "RAW photo, man, high detailed skin, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
    generator = torch.Generator(device="cuda").manual_seed(1337)
    image = pipe(
        prompt=prompt, 
        image=rescaled_face, 
        strength=0.4, 
        guidance_scale=7.5, 
        generator=generator).images[0]
    return image

def postScale(reference_image, target_image):
    reference_size = reference_image.size
    resized_target_image = target_image.resize(reference_size, Image.BICUBIC)
    return resized_target_image

def insert_image(reference_image, small_image, top, left):
    small_width, small_height = small_image.size
    box = (left, top, left + small_width, top + small_height)
    reference_image.paste(small_image, box)
    return reference_image

def createMask(image):
    if isinstance(image, str):
        # If the input is a file path, open the image using PIL
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        # If the input is already a PIL Image object, use it directly
        img = image
    else:
        raise ValueError("Invalid input. Please provide an image file path or a PIL Image object.")

    src = np.array(img)

    mask = np.zeros_like(src)

    height, width, _ = src.shape

    rect_width = int(width * 0.8)
    rect_height = int(height * 0.8)
    start_x = int((width - rect_width) / 2)
    start_y = int((height - rect_height) / 2)
    end_x = start_x + rect_width
    end_y = start_y + rect_height

    cv2.rectangle(mask, (start_x, start_y), (end_x, end_y), (255, 255, 255), thickness=-1)

    border_size = int(width * 0.1)
    border_color = (0, 0, 0)
    mask[:border_size, :] = border_color  # Oben
    mask[-border_size:, :] = border_color  # Unten
    mask[:, :border_size] = border_color  # Links
    mask[:, -border_size:] = border_color  # Rechts

    mask_blur = cv2.GaussianBlur(mask, (31, 31), 0)

    cv2.imwrite('face_mask.jpg', mask_blur)

    return mask_blur

def combine_faces(orig_face, img_face_post, mask):
    src1 = np.array(orig_face)
    src2 = np.array(img_face_post)
    mask1 = np.array(mask)
    mask1 = mask1 / 255
    dst = src2 * mask1 + src1 * (1 - mask1)
    smooth_face = Image.fromarray(dst.astype(np.uint8))
    return  smooth_face

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "philz1337/rv3.0", 
    torch_dtype=torch.float16
    ).to(
    "cuda"
)


def fixFace(image_link, save=False):
    orig_image = Image.open(image_link)
    
    face_image, top, left = find_face(image_link)
    
    rescaled_face = preScale(face_image)
    
    image = faceImg2img(rescaled_face)
    
    img_face_post = postScale(face_image, image)
    
    mask = createMask(face_image)

    img_smooth_face = combine_faces(face_image, img_face_post, mask)

    new_image = insert_image(orig_image, img_smooth_face, top, left)
   
    if save:
        face_image.save("0_facebox.png")
        rescaled_face.save("1_rescaled_face.png")
        image.save("2_img2img_result.png")
        img_face_post.save("3_face_old_result.png")
        img_smooth_face.save('4_combined_face.jpg')
        new_image.save("5_new_image.png")

fixFace("pre.png", True)
