import math
from os import PathLike
from pathlib import Path
from typing import Optional, List, Union
from safetensors import safe_open
import torch
from pydantic import BaseModel, Field
from diffusers import QwenImageEditPlusPipeline
import time
from PIL import Image

import json

from dotenv import load_dotenv

from schemas.custom_types import BFloatTensor, IntTensor
load_dotenv()

from logger_config import logger

from config import Settings


class EmbeddedPrompting(BaseModel):
    prompt_embeds: BFloatTensor
    prompt_embeds_mask: Optional[IntTensor] = None


class TextPrompting(BaseModel):
    prompt: str = Field(alias="positive")
    negative_prompt: Optional[str] = Field(default=" ", alias="negative")


class QwenImageEditPlusModule:
    """Qwen module for image editing operations using QwenImageEditPlusPipeline (2511)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipe: Optional[QwenImageEditPlusPipeline] = None
        self.device = f"cuda:{settings.qwen_gpu}" if torch.cuda.is_available() else "cpu"
        self.dtype = self._resolve_dtype(settings.dtype)
        self.gpu_index = settings.qwen_gpu

        self._empty_image = Image.new('RGB', (1024, 1024))
        self.edit_model_path = settings.qwen_edit_model_path
        self.edit_model_lora_path = settings.qwen_edit_lora_path
        self.edit_model_lora_ckpt = settings.qwen_edit_lora_ckpt
        self.prompt_path = settings.qwen_edit_prompt_path
        self.prompting = self._set_prompting()

        self.pipe_config = {
            "num_inference_steps": settings.num_inference_steps,
            "true_cfg_scale": settings.true_cfg_scale,
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
        }

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        resolved = mapping.get(dtype.lower(), torch.bfloat16)
        if not torch.cuda.is_available() and resolved in {torch.float16, torch.bfloat16}:
            return torch.float32
        return resolved

    async def startup(self) -> None:
        """Initialize the Qwen pipeline."""
        logger.info("Initializing QwenImageEditPlusModule...")
        await self._load_pipeline()
        logger.success("QwenImageEditPlusModule ready.")

    async def shutdown(self) -> None:
        """Shutdown the pipeline and free resources."""
        if self.pipe:
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
        self.pipe = None
        logger.info("QwenImageEditPlusModule closed.")

    def is_ready(self) -> bool:
        """Check if pipeline is loaded and ready."""
        return self.pipe is not None

    async def _load_pipeline(self) -> None:
        """Load the QwenImageEditPlusPipeline."""
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.gpu_index)
            except Exception as err:
                logger.warning(f"Failed to set CUDA device ({self.gpu_index}): {err}")

        t1 = time.time()

        # Load pipeline directly from Qwen/Qwen-Image-Edit-2511
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.edit_model_path,
            torch_dtype=self.dtype
        )

        if self.edit_model_lora_path and self.edit_model_lora_ckpt:
            self.pipe.load_lora_weights(
                self.edit_model_lora_path,
                weight_name=self.edit_model_lora_ckpt
            )
            self.pipe.fuse_lora()
            
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=None)

        load_time = time.time() - t1
        logger.success(f"Qwen pipeline ready (loading: {load_time:.2f}s). Loaded on {self.device} with dtype={self.dtype}.")

    def _set_text_prompting(self, path: Optional[PathLike] = None) -> TextPrompting:
        path = path or self.prompt_path
        with open(path, "r") as f:
            edit_prompt = TextPrompting.model_validate_json(json.dumps(json.load(f)))
            return edit_prompt

    def _set_embedded_prompting(self, path: Optional[PathLike] = None) -> EmbeddedPrompting:
        path = path or self.prompt_path
        with safe_open(path, framework="pt", device=self.device) as f:
            tensors = {key: f.get_tensor(key) for key in f.keys()}
            embedding = EmbeddedPrompting(**tensors)
        return embedding

    def _set_prompting(self, path: Optional[PathLike] = None) -> TextPrompting | EmbeddedPrompting:
        path = Path(path or self.prompt_path)
        if path.suffix == ".safetensors":
            return self._set_embedded_prompting(path)
        else:
            return self._set_text_prompting(path)

    def _prepare_input_image(self, image: Image.Image, megapixels: float = 1.0) -> Image.Image:
        total = int(megapixels * 1024 * 1024)
        scale_by = math.sqrt(total / (image.width * image.height))
        width = round(image.width * scale_by)
        height = round(image.height * scale_by)
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def _run_model_pipe(
        self,
        image: Union[Image.Image, List[Image.Image]],
        prompt: str,
        negative_prompt: str = " ",
        seed: Optional[int] = None,
        **kwargs
    ):
        """Run the pipeline with the given inputs."""
        inputs = {
            "image": image if isinstance(image, list) else [image],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            **self.pipe_config,
            **kwargs,
        }

        if seed is not None:
            inputs["generator"] = torch.Generator(device=self.device).manual_seed(seed)

        with torch.inference_mode():
            result = self.pipe(**inputs)

        return result

    def _run_edit_pipe(
        self,
        prompt_image: Union[Image.Image, List[Image.Image]],
        prompt: str,
        negative_prompt: str = " ",
        seed: Optional[int] = None,
        **kwargs
    ):
        """Run edit pipeline with prepared images."""
        if isinstance(prompt_image, list):
            prepared_images = [self._prepare_input_image(img) for img in prompt_image]
        else:
            prepared_images = [self._prepare_input_image(prompt_image)]

        logger.info(f"Prompt image(s) prepared for editing")
        logger.info(f"Prompt: {prompt}")

        return self._run_model_pipe(
            image=prepared_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            **kwargs
        )

    def edit_image(
        self,
        prompt_image: Union[Image.Image, List[Image.Image]],
        seed: int,
        prompt: Optional[str] = None
    ) -> Image.Image:
        """
        Edit the image using Qwen Edit 2511.

        Args:
            prompt_image: The prompt image(s) to edit. Can be single image or list of images.
            seed: Random seed for generation.
            prompt: Optional prompt override.

        Returns:
            The edited image.
        """
        if self.pipe is None:
            logger.error("Edit Model is not loaded")
            raise RuntimeError("Edit Model is not loaded")

        try:
            start_time = time.time()

            # Get prompting configuration
            prompting = self.prompting.model_dump()
            if prompt:
                prompting["prompt"] = prompt

            edit_prompt = prompting.get("prompt", "")
            negative_prompt = prompting.get("negative_prompt", " ")

            # Run the edit pipe
            result = self._run_edit_pipe(
                prompt_image=prompt_image,
                prompt=edit_prompt,
                negative_prompt=negative_prompt,
                seed=seed
            )

            generation_time = time.time() - start_time

            image_edited = result.images[0]

            logger.success(f"Edited image generated in {generation_time:.2f}s, Size: {image_edited.size}, Seed: {seed}")

            return image_edited

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise e


# Alias for backward compatibility
QwenEditModule = QwenImageEditPlusModule
