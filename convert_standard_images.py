"""
Convert standard test images to grayscale PGM format
Place source images (lena.png, baboon.png, airplane.png, peppers.png) 
in data/input/standard/ before running this script
"""
import cv2
import os
import numpy as np

input_folder = "data/input/standard"
output_folder = "data/input/standard"

# Standard test images
images = {
    "lena": ["lena.png", "lena.jpg", "lena.jpeg", "lena.bmp", "lena.tiff"],
    "baboon": ["baboon.png", "baboon.jpg", "mandrill.png", "mandrill.jpg"],
    "airplane": ["airplane.png", "airplane.jpg", "f16.png", "f16.jpg"],
    "peppers": ["peppers.png", "peppers.jpg", "pepper.png", "pepper.jpg"]
}

print("Converting standard images to grayscale PGM format...\n")

converted = 0
for base_name, possible_names in images.items():
    found = False
    for name in possible_names:
        path = os.path.join(input_folder, name)
        if os.path.exists(path):
            print(f"Found: {name}")
            img = cv2.imread(path)
            
            if img is None:
                print(f"  ❌ Cannot read {name}")
                continue
            
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            # Resize to 512x512 (standard size for these images)
            if gray.shape != (512, 512):
                gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
                print(f"  Resized to 512x512")
            
            # Save as PGM
            output_path = os.path.join(output_folder, f"{base_name}.pgm")
            cv2.imwrite(output_path, gray)
            print(f"  ✅ Saved: {base_name}.pgm ({gray.shape})")
            print(f"     Stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}\n")
            
            converted += 1
            found = True
            break
    
    if not found:
        print(f"❌ Not found: {base_name} (tried: {', '.join(possible_names)})\n")

print(f"{'='*60}")
print(f"Converted {converted}/4 images successfully")
print(f"{'='*60}")

if converted > 0:
    print("\nYou can now use these images in the application!")
    print("Select 'standard' folder in the GUI.")
