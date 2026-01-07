from PIL import Image
from .shadow import ShadowDetector
import os

# Initialize detector
detector = ShadowDetector()

# Load an image with removed background
image = Image.open("/Users/congdc/Project/Axava-Labs/ping001/assets/10.jpg")

# Simple detection
is_symmetric = detector.detect(image)
print(f"Object is symmetric: {is_symmetric}")

# Detection with scores
is_symmetric, ssim_score, psnr_score = detector.detect_with_scores(image)
print(f"Symmetric: {is_symmetric}, SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.2f}")

# Test all images in folder

folder_path = "/Users/congdc/Project/Axava-Labs/ping001/assets"
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        is_sym, ssim, psnr = detector.detect_with_scores(img)
        print(f"{filename}: Symmetric={is_sym}, SSIM={ssim:.4f}, PSNR={psnr:.2f}")
