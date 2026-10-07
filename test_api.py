from pathlib import Path
import io
import requests
from PIL import Image, ImageDraw

API_URL = "http://127.0.0.1:5000"

# Create a simple local image for API connectivity testing.
# It is only a smoke test; CIFAR-10 accuracy is evaluated using the test set.
img = Image.new("RGB", (32, 32), "white")
draw = ImageDraw.Draw(img)
draw.rectangle((6, 6, 26, 26), outline="black", width=2)

buffer = io.BytesIO()
img.save(buffer, format="PNG")
buffer.seek(0)

response = requests.post(
    f"{API_URL}/predict",
    files={"image": ("test.png", buffer.getvalue(), "image/png")},
    timeout=60,
)

print("HTTP status:", response.status_code)
print(response.json())
