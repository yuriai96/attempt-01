from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional

from PIL import Image
import pyspz
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import (
    GenerateRequest,
    GenerateResponse,
    TrellisParams,
    TrellisRequest,
    TrellisResult,
)
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.rmbg_manager import BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.utils import (
    secure_randint,
    set_random_seed,
    decode_image,
    to_png_base64,
    save_files,
)
from modules.judge.render import render_image_combine, combine_images4
from modules.judge.score import judge_3d_duel, calculate_ssim, calculate_psnr

class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BackgroundRemovalService(settings)
        self.trellis = TrellisService(settings)
        self.estimate_time_gen3d = 8
        self.estimate_time_valid = 3

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()

        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()

        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""

        temp_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_imge_bytes = buffer.getvalue()
        await self.generate_from_upload(temp_imge_bytes, seed=42)

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.

        Args:
            image_bytes: Raw image bytes from uploaded file

        Returns:
            PLY file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create request
        request = GenerateRequest(
            prompt_image=image_base64, prompt_type="image", seed=seed
        )

        # Generate
        response = await self.generate_gs(request)

        # Return binary PLY
        if not response.ply_file_base64:
            raise ValueError("PLY generation failed")

        return response.ply_file_base64  # bytes
    

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.

        Args:
            request: Generation request with prompt and settings

        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)
        
        org_img_edit = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=request.seed,
                prompt="Turn the background to a uniform solid color. Keep the main subject unchanged: same scale, orientation, camera perspective, lighting direction, shadows, and original color palette. Only the background should be modified.",
            )
        original_image_edit_without_background = self.rmbg.remove_background(org_img_edit)

        # 1. Edit the image using Qwen Edit
        image_edited_1 = self.qwen_edit.edit_image(
            prompt_image=org_img_edit,
            seed=request.seed,
            prompt="Left side view. The object stays fixed in space; only the camera rotates 90 degrees to the left. Preserve the original scale, proportions, lighting direction, shadows, camera distance, and color palette. Deep depth of field, everything in sharp focus.",
        )
        
        image_edited_2 = self.qwen_edit.edit_image(
            prompt_image=org_img_edit,
            seed=request.seed,
            prompt="Right side view. The object stays fixed in space; only the camera rotates 90 degrees to the left. Preserve the original scale, proportions, lighting direction, shadows, camera distance, and color palette. Deep depth of field, everything in sharp focus.",
        )
        
        
        image_edited_3 = self.qwen_edit.edit_image(
            prompt_image=org_img_edit,
            seed=request.seed,
            prompt="Back view. The object stays fixed in space; only the camera rotates 90 degrees to the left. Preserve the original scale, proportions, lighting direction, shadows, camera distance, and color palette. Deep depth of field, everything in sharp focus.",
        )
        

        # 2. Remove background
        image_without_background_1 = self.rmbg.remove_background(image_edited_1)
        image_without_background_2 = self.rmbg.remove_background(image_edited_2)
        image_without_background_3 = self.rmbg.remove_background(image_edited_3)
        
        original_image_without_background = self.rmbg.remove_background(image)
        

        trellis_result: Optional[TrellisResult] = None
        idx_best_result = 0
        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params

        # 3. Generate the 3D model
        s0 = time.time()
        trellis_result_1 = self.trellis.generate(
            TrellisRequest(
                images=[
                    original_image_edit_without_background,
                    image_without_background_1,
                    image_without_background_2,
                    image_without_background_3,
                ],
                seed=request.seed,
                params=trellis_params,
            )
        )
        self.estimate_time_gen3d = time.time() - s0
        # render_1.save("render_1.png")
        render_1 = render_image_combine(trellis_result_1.ply_file)
        trellis_result = trellis_result_1
        best_render = render_1
        idx_best_result = 1
        
        if time.time() - t1 + self.estimate_time_gen3d + self.estimate_time_valid < 32:
            s0 = time.time()
            trellis_result_2 = self.trellis.generate(
                TrellisRequest(
                    images=[
                        original_image_without_background,
                    ],
                    seed=request.seed,
                    params=trellis_params,
                )
            )
            render_2 = render_image_combine(trellis_result_2.ply_file)
            # render_2.save("render_2.png")
            self.estimate_time_gen3d = max(self.estimate_time_gen3d, time.time() - s0)
           
            s1 = time.time()
            winner, avg_penalty_left, avg_penalty_right, issues = await judge_3d_duel(render_1, render_2, image)
            self.estimate_time_valid = max(self.estimate_time_valid, time.time() - s1)
            print(f"Winner: {winner}, Left penalty: {avg_penalty_left}, Right penalty: {avg_penalty_right}, Issues: {issues}")
            if avg_penalty_left < avg_penalty_right:
                pass
            elif avg_penalty_left > avg_penalty_right:
                trellis_result = trellis_result_2
                best_render = render_2
                idx_best_result = 2
            else:
                if len(trellis_result_1.ply_file) < len(trellis_result_2.ply_file):
                    pass
                else:
                    trellis_result = trellis_result_2
                    best_render = render_2
                    idx_best_result = 2

        if time.time() - t1 + self.estimate_time_gen3d + self.estimate_time_valid < 32:
            print(f"Generate result 3")
            s0 = time.time()
            trellis_result_3 = self.trellis.generate(
                TrellisRequest(
                    images=[
                        original_image_edit_without_background,
                        image_without_background_3,
                    ],
                    seed=request.seed,
                    params=trellis_params,
                )
            )
            
            render_3 = render_image_combine(trellis_result_3.ply_file)
            # render_3.save("render_3.png")
            self.estimate_time_gen3d = max(self.estimate_time_gen3d, time.time() - s0)

            s1 = time.time()
            winner, avg_penalty_left, avg_penalty_right, issues = await judge_3d_duel(render_3, best_render, image)
            self.estimate_time_valid = max(self.estimate_time_valid, time.time() - s1)
            print(f"Winner: {winner}, Left penalty: {avg_penalty_left}, Right penalty: {avg_penalty_right}, Issues: {issues}")
            if avg_penalty_left < avg_penalty_right:
                trellis_result = trellis_result_3
                best_render = render_3
                idx_best_result = 3
            elif avg_penalty_left > avg_penalty_right:
                pass
            else:
                if len(trellis_result_3.ply_file) < len(trellis_result_2.ply_file):
                    trellis_result = trellis_result_3
                    best_render = render_3
                    idx_best_result = 3
                else:
                    pass

        print(f"Best result index: {idx_best_result}")

        # Save generated files
        if self.settings.save_generated_files:
            save_files(trellis_result, image_edited_1, image_without_background_1)

        # Convert to PNG base64 for response (only if needed)
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited_1)
            image_without_background_base64 = to_png_base64(image_without_background_1)

        t2 = time.time()
        generation_time = t2 - t1

        logger.info(f"Total generation time: {generation_time} seconds")
        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result.ply_file if trellis_result else None,
            image_edited_file_base64=image_edited_base64
            if self.settings.send_generated_files
            else None,
            image_without_background_file_base64=image_without_background_base64
            if self.settings.send_generated_files
            else None,
        )
        return response
