"""
Download standard test images from USC-SIPI Image Database
"""
import urllib.request
import os

output_folder = "data/input/standard"
os.makedirs(output_folder, exist_ok=True)

# Standard test images URLs (grayscale, 512x512)
images = {
    "lena.pgm": "https://sipi.usc.edu/database/download.php?vol=misc&img=5.1.09",
    "baboon.pgm": "https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03",
    "airplane.pgm": "https://sipi.usc.edu/database/download.php?vol=misc&img=7.1.03",
    "peppers.pgm": "https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.07"
}

print("Downloading standard test images...\n")

for filename, url in images.items():
    output_path = os.path.join(output_folder, filename)
    try:
        print(f"Downloading {filename}...", end=" ")
        urllib.request.urlretrieve(url, output_path)
        print("✅")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n✅ Done! You can now use 'standard' folder in the application.")
