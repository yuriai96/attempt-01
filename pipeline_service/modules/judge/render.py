import sys
import torch 
import time
from pathlib import Path
from PIL import Image
import io
import numpy as np
import requests 

def load_ply_to_buffer(ply_path: str) -> io.BytesIO:
    buffer = io.BytesIO()
    with open(ply_path, "rb") as f:
        buffer.write(f.read())
    buffer.seek(0)
    return buffer

def render_image_combine(ply_file: bytes) -> Image.Image:
    s0 = time.time()
    
    # Call the render service API
    response = requests.post(
        "http://localhost:8000/render",
        files={"file": ("content.ply", ply_file, "application/octet-stream")}
    )
    
    if response.status_code == 200:
        # Load the returned PNG bytes into an Image
        grid = Image.open(io.BytesIO(response.content))
        print(f" Render image combine done in {time.time() - s0:.2f} s")
        return grid
    else:
        print(f"Failed to render image: HTTP {response.status_code} - {response.text}")
        return None
    
def combine_images4(
    images: list[Image.Image],
    img_width: int = 518,
    img_height: int = 518,
    gap: int = 5,
    background_color: str = "black"
) -> Image.Image:
    """
    Args:
        images: List of 4 PIL Images to combine
        img_width: Width of each individual image (default: 518)
        img_height: Height of each individual image (default: 518)
        gap: Gap between images in pixels (default: 5)
        background_color: Background/gap color (default: "black")
    
    Returns:
        Combined PIL Image with dimensions (img_width*2 + gap, img_height*2 + gap)
        For default values: 1041 x 1041 pixels
    """
    if len(images) != 4:
        raise ValueError(f"Expected 4 images, got {len(images)}")
    
    # Calculate final dimensions
    row_width = img_width * 2 + gap
    column_height = img_height * 2 + gap
    
    # Create canvas with background color
    combined_image = Image.new("RGB", (row_width, column_height), color=background_color)
    
    # Resize images if needed
    resized_images = []
    for img in images:
        if img.size != (img_width, img_height):
            img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        resized_images.append(img)
    
    # Paste images into grid positions
    # Position (0, 0): Top-left - View at 22.5°
    combined_image.paste(resized_images[0], (0, 0))
    
    # Position (1, 0): Top-right - View at 112.5°
    combined_image.paste(resized_images[1], (img_width + gap, 0))
    
    # Position (0, 1): Bottom-left - View at 202.5°
    combined_image.paste(resized_images[2], (0, img_height + gap))
    
    # Position (1, 1): Bottom-right - View at 292.5°
    combined_image.paste(resized_images[3], (img_width + gap, img_height + gap))
    
    return combined_image
    
if __name__ == "__main__":
    import argparse

    
    parser = argparse.ArgumentParser(description='Render 2x2 grid from a single PLY file')
    parser.add_argument('--input_ply', type=str, required=True, help='Path to the input PLY file')
    parser.add_argument('--output_image', type=str, required=True, help='Path to save the output image')
    
    args = parser.parse_args()
    
    input_ply_path = Path(args.input_ply)
    buffer = load_ply_to_buffer(input_ply_path)
    ply_file=buffer.getvalue()
    
    
    img_result = render_image_combine(ply_file)
    
    if img_result:
        output_image_path = args.output_image
