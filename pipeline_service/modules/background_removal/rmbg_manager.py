from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, resized_crop

from config import Settings
from logger_config import logger


class BaseBackgroundRemover(ABC):
    """Abstract base class for background removal models."""
    
    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model."""
        pass
    
    @abstractmethod
    def inference(self, image: Image.Image) -> Image.Image:
        """Run inference on a single image."""
        pass
    
    @abstractmethod
    def inference_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        """Run inference on a batch of images."""
        pass


class BiRefNetRemover(BaseBackgroundRemover):
    """BiRefNet background removal model."""
    
    def __init__(self, model_id: str = "ZhengPeng7/BiRefNet"):
        self.model_id = model_id
        self.model = None
        self.device = None
        self.transform = None
        self.image_size = (1024, 1024)
    
    def load(self, device: torch.device) -> None:
        from transformers import AutoModelForImageSegmentation
        
        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        torch.set_float32_matmul_precision('high')
        self.model.to(device)
        self.model.eval()
        self.model.half()
        
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.success(f"BiRefNet model loaded from {self.model_id}")
    
    def unload(self) -> None:
        self.model = None
        self.transform = None
        logger.info("BiRefNet model unloaded.")
    
    def inference(self, image: Image.Image) -> Image.Image:
        if self.model is None:
            raise RuntimeError("BiRefNet model not loaded.")
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            rgb_image = image.convert("RGB")
        else:
            rgb_image = image
        
        original_size = image.size
        
        # Transform and predict
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device).half()
        
        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()
        
        # Process mask
        pred = preds[0].squeeze()
        mask_pil = transforms.ToPILImage()(pred)
        mask = mask_pil.resize(original_size, Image.Resampling.LANCZOS)
        
        # Apply mask to original image
        result = rgb_image.copy()
        result.putalpha(mask)
        
        return result
    
    def inference_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        if self.model is None:
            raise RuntimeError("BiRefNet model not loaded.")
        
        if not images:
            return []
        
        results = []
        original_sizes = []
        input_tensors = []
        
        for image in images:
            if image.mode != "RGB":
                rgb_image = image.convert("RGB")
            else:
                rgb_image = image
            
            original_sizes.append(image.size)
            input_tensors.append(self.transform(rgb_image))
        
        # Stack tensors for batch processing
        batch_tensor = torch.stack(input_tensors).to(self.device).half()
        
        with torch.no_grad():
            preds = self.model(batch_tensor)[-1].sigmoid().cpu()
        
        # Process each prediction
        for i, pred in enumerate(preds):
            pred_squeezed = pred.squeeze()
            mask_pil = transforms.ToPILImage()(pred_squeezed)
            mask = mask_pil.resize(original_sizes[i], Image.Resampling.LANCZOS)
            
            # Get original RGB image
            if images[i].mode != "RGB":
                rgb_image = images[i].convert("RGB")
            else:
                rgb_image = images[i].copy()
            
            rgb_image.putalpha(mask)
            results.append(rgb_image)
        
        return results


class BEN2Remover(BaseBackgroundRemover):
    """BEN2 background removal model."""
    
    def __init__(self, model_id: str = "PramaLLC/BEN2"):
        self.model_id = model_id
        self.model = None
        self.device = None
    
    def load(self, device: torch.device) -> None:
        from ben2 import BEN_Base
        
        self.device = device
        self.model = BEN_Base.from_pretrained(self.model_id)
        self.model.to(device).eval()
        
        logger.success(f"BEN2 model loaded from {self.model_id}")
    
    def unload(self) -> None:
        self.model = None
        logger.info("BEN2 model unloaded.")
    
    def inference(self, image: Image.Image) -> Image.Image:
        if self.model is None:
            raise RuntimeError("BEN2 model not loaded.")
        
        with torch.no_grad():
            foreground = self.model.inference(image, refine_foreground=False)
        
        return foreground
    
    def inference_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        if self.model is None:
            raise RuntimeError("BEN2 model not loaded.")
        
        if not images:
            return []
        
        with torch.no_grad():
            foregrounds = self.model.inference(images, refine_foreground=False)
        
        # Handle single image return
        if not isinstance(foregrounds, list):
            foregrounds = [foregrounds]
        
        return foregrounds


class BackgroundRemovalService:
    """Background removal service supporting multiple models."""
    
    SUPPORTED_MODELS = {
        "birefnet": BiRefNetRemover,
        "ben2": BEN2Remover,
    }
    
    def __init__(self, settings: Settings):
        """
        Initialize the BackgroundRemovalService.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Set padding percentage, output size
        self.padding_percentage = self.settings.padding_percentage
        self.output_size = self.settings.output_image_size
        self.limit_padding = self.settings.limit_padding
        
        # Set device
        self.device = torch.device(
            f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"
        )
        
        # Model settings
        self.model_type: Literal["birefnet", "ben2"] = settings.rmbg_model
        self.remover: Optional[BaseBackgroundRemover] = None

    async def startup(self) -> None:
        """Startup the BackgroundRemovalService."""
        logger.info(f"Loading background removal model: {self.model_type}...")
        
        try:
            if self.model_type == "birefnet":
                self.remover = BiRefNetRemover(self.settings.birefnet_model_id)
            elif self.model_type == "ben2":
                self.remover = BEN2Remover()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.remover.load(self.device)
            logger.success(f"Background removal model ({self.model_type}) loaded.")
        except Exception as e:
            logger.error(f"Error loading background removal model: {e}")
            raise RuntimeError(f"Error loading background removal model: {e}")

    async def shutdown(self) -> None:
        """Shutdown the BackgroundRemovalService."""
        if self.remover is not None:
            self.remover.unload()
            self.remover = None
        logger.info("BackgroundRemovalService closed.")

    def ensure_ready(self) -> None:
        """Ensure the BackgroundRemovalService is ready."""
        if self.remover is None:
            raise RuntimeError("Background removal model not initialized.")

    def switch_model(self, model_type: Literal["birefnet", "ben2"]) -> None:
        """
        Switch to a different background removal model.
        
        Args:
            model_type: The model to switch to ("birefnet" or "ben2")
        """
        if model_type == self.model_type and self.remover is not None:
            logger.info(f"Already using {model_type}, no switch needed.")
            return
        
        logger.info(f"Switching from {self.model_type} to {model_type}...")
        
        # Unload current model
        if self.remover is not None:
            self.remover.unload()
        
        # Load new model
        if model_type == "birefnet":
            self.remover = BiRefNetRemover(self.settings.birefnet_model_id)
        elif model_type == "ben2":
            self.remover = BEN2Remover()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.remover.load(self.device)
        self.model_type = model_type
        logger.success(f"Switched to {model_type}.")

    def remove_background(self, image: Image.Image) -> Image.Image:
        """
        Remove the background from a single image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            PIL Image with background removed (RGBA)
        """
        self.ensure_ready()
        
        try:
            t1 = time.time()
            
            # Check if the image already has alpha channel with transparency
            if image.mode == "RGBA":
                alpha = np.array(image)[:, :, 3]
                if not np.all(alpha == 255):
                    # Image already has transparency, just crop and resize
                    output = self._crop_and_resize(image)
                    removal_time = time.time() - t1
                    logger.success(
                        f"Background remove (skip) - Time: {removal_time:.2f}s - "
                        f"OutputSize: {output.size}"
                    )
                    return output

            # Run inference
            foreground = self.remover.inference(image)
            
            # Crop and resize the result
            output = self._crop_and_resize(foreground)
            
            removal_time = time.time() - t1
            logger.success(
                f"Background remove ({self.model_type}) - Time: {removal_time:.2f}s - "
                f"OutputSize: {output.size} - InputSize: {image.size}"
            )

            return output
            
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            return image

    def remove_background_batch(self, images: List[Image.Image]) -> List[Image.Image]:
        """
        Remove the background from multiple images using batch inference.
        
        Args:
            images: List of input PIL Images
            
        Returns:
            List of PIL Images with background removed (RGBA)
        """
        self.ensure_ready()
        
        if not images:
            return []
        
        try:
            t1 = time.time()
            
            results = []
            images_to_process = []
            indices_to_process = []
            
            # Check which images need processing
            for i, image in enumerate(images):
                if image.mode == "RGBA":
                    alpha = np.array(image)[:, :, 3]
                    if not np.all(alpha == 255):
                        # Image already has transparency
                        results.append((i, self._crop_and_resize(image)))
                        continue
                
                images_to_process.append(image)
                indices_to_process.append(i)
            
            # Batch process images that need background removal
            if images_to_process:
                foregrounds = self.remover.inference_batch(images_to_process)
                
                for idx, fg in zip(indices_to_process, foregrounds):
                    results.append((idx, self._crop_and_resize(fg)))
            
            # Sort results by original index
            results.sort(key=lambda x: x[0])
            output_images = [img for _, img in results]
            
            removal_time = time.time() - t1
            logger.success(
                f"Background remove batch ({self.model_type}) - "
                f"Time: {removal_time:.2f}s - Count: {len(images)}"
            )

            return output_images
            
        except Exception as e:
            logger.error(f"Error removing background batch: {e}")
            return images

    def _crop_and_resize(self, image: Image.Image) -> Image.Image:
        """
        Crop the image to the bounding box of non-transparent pixels and resize.
        
        Args:
            image: PIL Image with RGBA mode
            
        Returns:
            Cropped and resized PIL Image
        """
        # Ensure RGBA mode
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        
        # Get alpha channel
        img_array = np.array(image)
        alpha = img_array[:, :, 3]
        
        # Find bounding box of non-transparent pixels
        bbox_indices = np.argwhere(alpha > 0.8 * 255)
        
        if len(bbox_indices) == 0:
            # No foreground found, return resized original
            return image.resize(self.output_size, Image.Resampling.LANCZOS)
        
        # Get bounding box
        h_min, w_min = bbox_indices.min(axis=0)
        h_max, w_max = bbox_indices.max(axis=0)
        
        width = w_max - w_min
        height = h_max - h_min
        
        # Calculate center and padded size
        center_h = (h_max + h_min) / 2
        center_w = (w_max + w_min) / 2
        size = max(width, height)
        padded_size = int(size * (1 + self.padding_percentage))
        
        # Calculate crop coordinates
        top = int(center_h - padded_size // 2)
        left = int(center_w - padded_size // 2)
        bottom = int(center_h + padded_size // 2)
        right = int(center_w + padded_size // 2)
        
        if self.limit_padding:
            top = max(0, top)
            left = max(0, left)
            bottom = min(alpha.shape[0], bottom)
            right = min(alpha.shape[1], right)
        
        # Convert to tensor for crop operation
        img_tensor = transforms.ToTensor()(image)
        
        # Crop and resize
        output_tensor = resized_crop(
            img_tensor,
            top=top,
            left=left,
            height=bottom - top,
            width=right - left,
            size=self.output_size,
            antialias=True
        )
        
        return to_pil_image(output_tensor)

