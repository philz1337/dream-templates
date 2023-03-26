import base64
import requests
import sys


def gen(output_fn, **kwargs):
    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    try:
        datauri = data["output"][0]
        base64_encoded_data = datauri.split(",")[1]
        data = base64.b64decode(base64_encoded_data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data['logs'])
        sys.exit(1)
 
    with open(output_fn, "wb") as f:
        f.write(data)


def main():
    gen(
        "sample.controlnet_txt2img.png",
        prompt="painting of cjw by van gogh",
        control_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/converted/control_vermeer_scribble.png",
        seed=42,
    )
    gen(
        "sample.txt2img.png",
        prompt="painting of cjw by van gogh",
        seed=42
    )
    gen(
        "sample.controlnet_img2img.png",
        prompt="painting of cjw by van gogh",
        control_image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/converted/control_vermeer_scribble.png",
        image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/human_512x512.png",
        seed=42
    )
    gen(
        "sample.img2img.png",
        prompt="painting of cjw by van gogh",
        image="https://huggingface.co/takuma104/controlnet_dev/resolve/main/gen_compare/control_images/vermeer_512x512.png",
        seed=42
    )


if __name__ == "__main__":
    main()
