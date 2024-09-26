import requests
from PIL import Image

# Hugging Face API token
API_TOKEN = "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"

# Model endpoint URL
model_url = "https://api-inference.huggingface.co/models/jinhybr/OCR-Donut-CORD"

# Open and prepare the image
image = Image.open("receipt2.png").convert("RGB")

# Convert image to bytes
image_bytes = image.tobytes()

# Set up headers for the API call
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/octet-stream",
}

# Make the API call
response = requests.post(model_url, headers=headers, data=image_bytes)

# Parse the response
result = response.json()

# Print extracted data
print(result)
