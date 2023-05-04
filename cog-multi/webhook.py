import requests


def fire_webhook(
    webhook_url, output_path, idx
):

    # Create the payload to be sent to the webhook
    payload = {
        "num_images": idx,
        "output_path": output_path
    }

    # Send the POST request to the webhook
    response = requests.post(webhook_url, json=payload, timeout=0.5)

    # Check the status code of the response
    if response.status_code == 200:
        print("Webhook fired successfully!")
    else:
        print(f"Error firing the webhook. Status code: {response.status_code}")
