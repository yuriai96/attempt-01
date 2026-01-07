from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Tuple, Optional

import sys

sys.path.append("/Users/congdc/Project/Axava-Labs/ping001/pipeline_service")  # to import config and logger_config

from config import Settings, settings
from logger_config import logger


class ShadowDetector:
    """
    Shadow detection module for analyzing object symmetry.
    
    This module takes an image with removed background and checks if the object
    is symmetric by comparing the left and right halves using SSIM and PSNR metrics.
    """
    
    def __init__(self, settings: Settings = settings):
        """
        Initialize the ShadowDetector.
        
        Args:
            settings: Configuration settings containing thresholds
        """
        self.ssim_threshold = settings.shadow_ssim_threshold
        self.psnr_threshold = settings.shadow_psnr_threshold
    
    def _get_object_bbox(self, image: Image.Image) -> Tuple[int, int, int, int]:
        """
        Get the bounding box of the object (non-transparent region).
        
        Args:
            image: RGBA image with transparent background
            
        Returns:
            Tuple of (left, top, right, bottom) coordinates
        """
        if image.mode != "RGBA":
            raise ValueError("Image must have RGBA mode with alpha channel")
        
        # Convert to numpy array
        img_array = np.array(image)
        alpha = img_array[:, :, 3]
        
        # Find non-transparent pixels
        non_transparent = np.argwhere(alpha > 0)
        
        if len(non_transparent) == 0:
            raise ValueError("No object found in image (all pixels are transparent)")
        
        # Get bounding box
        top = non_transparent[:, 0].min()
        bottom = non_transparent[:, 0].max()
        left = non_transparent[:, 1].min()
        right = non_transparent[:, 1].max()
        
        return left, top, right, bottom
    
    def _extract_object_halves(
        self, 
        image: Image.Image
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the left and right halves of the object.
        
        Args:
            image: RGBA image with transparent background
            
        Returns:
            Tuple of (left_rgb, right_rgb_flipped, left_mask, right_mask_flipped)
        """
        # Get object bounding box
        left, top, right, bottom = self._get_object_bbox(image)
        
        # Crop to object region
        img_array = np.array(image)
        object_region = img_array[top:bottom+1, left:right+1]
        
        # Get dimensions
        height, width = object_region.shape[:2]
        mid_x = width // 2
        
        # Split into left and right halves
        left_half = object_region[:, :mid_x]
        right_half = object_region[:, mid_x:2*mid_x]  # Ensure same width
        
        # Flip right half horizontally to match left orientation
        right_half_flipped = np.flip(right_half, axis=1)
        
        # Extract RGB and alpha channels
        left_rgb = left_half[:, :, :3]
        right_rgb_flipped = right_half_flipped[:, :, :3]
        left_mask = left_half[:, :, 3] > 0
        right_mask_flipped = right_half_flipped[:, :, 3] > 0
        
        return left_rgb, right_rgb_flipped, left_mask, right_mask_flipped
    
    def _compute_metrics(
        self,
        left_rgb: np.ndarray,
        right_rgb_flipped: np.ndarray,
        left_mask: np.ndarray,
        right_mask_flipped: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute SSIM and PSNR between the two halves, considering only overlapping object regions.
        
        Args:
            left_rgb: RGB values of left half
            right_rgb_flipped: Flipped RGB values of right half
            left_mask: Boolean mask of object pixels in left half
            right_mask_flipped: Flipped boolean mask of object pixels in right half
            
        Returns:
            Tuple of (ssim_score, psnr_score)
        """
        # Get overlapping mask (pixels that are part of object in both halves)
        overlap_mask = left_mask & right_mask_flipped
        
        if not np.any(overlap_mask):
            logger.warning("No overlapping object pixels found between halves")
            return 0.0, 0.0
        
        # Apply mask to both images - set non-overlapping pixels to 0
        left_masked = left_rgb.copy().astype(np.float64)
        right_masked = right_rgb_flipped.copy().astype(np.float64)
        
        # Create 3-channel mask
        mask_3ch = np.stack([overlap_mask] * 3, axis=-1)
        
        left_masked = left_masked * mask_3ch
        right_masked = right_masked * mask_3ch
        
        # Compute SSIM
        # Use multichannel for RGB images
        try:
            ssim_score = ssim(
                left_masked, 
                right_masked, 
                data_range=255.0,
                channel_axis=2
            )
        except Exception as e:
            logger.warning(f"SSIM computation failed: {e}")
            ssim_score = 0.0
        
        # Compute PSNR only on overlapping regions
        left_overlap = left_rgb[overlap_mask]
        right_overlap = right_rgb_flipped[overlap_mask]
        
        try:
            if len(left_overlap) > 0:
                psnr_score = psnr(
                    left_overlap.astype(np.float64), 
                    right_overlap.astype(np.float64), 
                    data_range=255.0
                )
            else:
                psnr_score = 0.0
        except Exception as e:
            logger.warning(f"PSNR computation failed: {e}")
            psnr_score = 0.0
        
        return ssim_score, psnr_score
    
    def detect(self, image: Image.Image) -> bool:
        """
        Detect if the object in the image has shadow (is symmetric).
        
        Args:
            image: PIL Image with removed background (RGBA format)
            
        Returns:
            True if object is symmetric (has "shadow"), False otherwise
        """
        try:
            # Ensure image is in RGBA mode
            if image.mode != "RGBA":
                logger.warning(f"Image mode is {image.mode}, converting to RGBA")
                image = image.convert("RGBA")
            
            # Extract object halves
            left_rgb, right_rgb_flipped, left_mask, right_mask_flipped = \
                self._extract_object_halves(image)
            
            # Compute metrics
            ssim_score, psnr_score = self._compute_metrics(
                left_rgb, right_rgb_flipped, left_mask, right_mask_flipped
            )
            
            logger.debug(f"Shadow detection - SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.2f}")
            
            # Check if both metrics exceed thresholds
            is_symmetric = (ssim_score >= self.ssim_threshold and 
                          psnr_score >= self.psnr_threshold)
            
            logger.info(
                f"Shadow detection result: {'symmetric' if is_symmetric else 'asymmetric'} "
                f"(SSIM: {ssim_score:.4f}/{self.ssim_threshold}, "
                f"PSNR: {psnr_score:.2f}/{self.psnr_threshold})"
            )
            
            return is_symmetric
            
        except Exception as e:
            logger.error(f"Shadow detection failed: {e}")
            return False
    
    def detect_with_scores(
        self, 
        image: Image.Image
    ) -> Tuple[bool, float, float]:
        """
        Detect if the object is symmetric and return the scores.
        
        Args:
            image: PIL Image with removed background (RGBA format)
            
        Returns:
            Tuple of (is_symmetric, ssim_score, psnr_score)
        """
        try:
            # Ensure image is in RGBA mode
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            
            # Extract object halves
            left_rgb, right_rgb_flipped, left_mask, right_mask_flipped = \
                self._extract_object_halves(image)
            
            # Compute metrics
            ssim_score, psnr_score = self._compute_metrics(
                left_rgb, right_rgb_flipped, left_mask, right_mask_flipped
            )
            
            # Check if both metrics exceed thresholds
            is_symmetric = (ssim_score >= self.ssim_threshold and 
                          psnr_score >= self.psnr_threshold)
            
            return is_symmetric, ssim_score, psnr_score
            
        except Exception as e:
            logger.error(f"Shadow detection failed: {e}")
            return False, 0.0, 0.0
