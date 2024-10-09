import requests
import base64

BASE = "http://127.0.0.1:5000/"

# Load image and convert to base64
with open("path_to_test_image.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Create payload with the image
test_image = {
    "file": "data:image/jpeg;base64," + encoded_image
}

# Send PUT request to the recommendation endpoint with the image
response = requests.put(BASE + "recommend", json=test_image)

# Print the response
print(response.json())
