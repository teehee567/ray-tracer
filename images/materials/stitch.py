import os
import sys
from PIL import Image

def stitch_images_horizontally(image_paths, output_path="stitched_image.jpg"):
    
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"Error opening image {path}: {e}")
    
    if not images:
        print("No valid images found. Exiting.")
        return
    
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    stitched_image = Image.new('RGB', (total_width, max_height))
    
    current_width = 0
    for img in images:
        stitched_image.paste(img, (current_width, 0))
        current_width += img.width
    
    stitched_image.save(output_path)
    print(f"Stitched image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:6]  # Take at most 5 images
        stitch_images_horizontally(image_paths)
    else:
        print("Usage: python stitch_images.py image1.jpg image2.jpg image3.jpg image4.jpg image5.jpg")
        print("Or manually edit the script to specify the image paths.")
        
        image_paths = [
            "raw/roughness/roughness_000.png",
            "raw/roughness/roughness_025.png",
            "raw/roughness/roughness_050.png",
            "raw/roughness/roughness_075.png",
            "raw/roughness/roughness_100.png",
        ]
        stitch_images_horizontally(image_paths, "my_stitched_panorama.jpg")
