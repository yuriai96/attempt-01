from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

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
from modules.judge.render import render_image_combine, combine_images4, rgba_to_rgb_white
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

    def _save_debug_image(self, image: Image.Image, index: int, task_id: str, suffix: str = "") -> None:
        """Save image in debug mode with index and task_id folder."""
        if not self.settings.debug:
            return
        
        # Create task-specific folder: debug/{task_id}/
        debug_dir = self.settings.output_dir / "debug" / task_id
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{index:02d}{('_' + suffix) if suffix else ''}.png"
        filepath = debug_dir / filename
        
        try:
            image.save(filepath, format="PNG")
            logger.debug(f"Debug: Saved {filepath}")
        except Exception as e:
            logger.error(f"Debug: Failed to save {filepath}: {e}")

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

        temp_image = Image.open("../assets/01.jpg")
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
        
        # Generate task_id for debug folder (unique per request)
        task_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        debug_index = 0

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)
        original_image_without_background = self.rmbg.remove_background(image)
        tmp_original_image_edit_with_white_bg = rgba_to_rgb_white(original_image_without_background)

        self._save_debug_image(image, debug_index, task_id, "input")
        debug_index += 1
        self._save_debug_image(original_image_without_background, debug_index, task_id, "original_rmbg")
        debug_index += 1

        # Within 5 seconds, generate variants and find the best one based on SSIM and PSNR
        best_ssim = -1
        best_psnr = -1
        num_attempt = 0
        max_attempt = 3
        list_prompt = ["Turn the background to a uniform solid color. Keep the main subject unchanged: same scale, orientation, camera perspective, lighting direction, shadows, and original color palette. Only the background should be modified.", "Turn the background to a uniform solid color", "Make the background a solid color. Keep camera pose, object color and object size"]
        best_org_img_edit = image
        best_original_image_edit_without_background = original_image_without_background
        
        start_time = time.time()
        current_seed = request.seed
        
        while time.time() - start_time < 5 and num_attempt <= max_attempt:
            
            # Generate new org_img_edit variant
            tmp_org_img_edit = self.qwen_edit.edit_image(
                prompt_image=image.copy(),
                seed=current_seed,
                prompt=list_prompt[num_attempt],
            )
            tmp_original_image_edit_without_background = self.rmbg.remove_background(tmp_org_img_edit)
            
            # Variant 2: simply resize original image to match comparison size
            compare_size = tmp_original_image_edit_without_background.size
            tmp_apply_white = image.copy().convert('RGB').resize(compare_size, Image.Resampling.LANCZOS)

            # Debug: Save org_img_edit variant
            self._save_debug_image(tmp_org_img_edit, debug_index, task_id, f"org_img_edit_attempt{num_attempt}")
            debug_index += 1
            self._save_debug_image(tmp_original_image_edit_without_background, debug_index, task_id, f"org_img_edit_rmbg_attempt{num_attempt}")
            debug_index += 1

            num_attempt += 1
            ssim_score = 0
            psnr_score = 0
            psnr_score_0 = round(calculate_ssim(tmp_org_img_edit, tmp_apply_white), 2)
            if max_attempt > 1 and psnr_score_0 < 39:
                # Variant 1: removed background image on white bg
                tmp_rmbg_white = Image.new('RGB', compare_size, (255, 255, 255))
                tmp_rmbg_white.paste(tmp_original_image_edit_without_background, mask=tmp_original_image_edit_without_background.split()[3] if tmp_original_image_edit_without_background.mode == 'RGBA' else None)
                
                
                
                # Debug: Save comparison images
                self._save_debug_image(tmp_rmbg_white, debug_index, task_id, f"compare_rmbg_white_attempt{num_attempt-1}")
                debug_index += 1
                self._save_debug_image(tmp_apply_white, debug_index, task_id, f"compare_apply_white_attempt{num_attempt-1}")
                debug_index += 1
                
                # Calculate SSIM and PSNR
                ssim_score = round(calculate_ssim(tmp_rmbg_white, tmp_apply_white), 2)
                psnr_score = round(calculate_psnr(tmp_rmbg_white, tmp_apply_white), 2)
                print(f"SSIM: {ssim_score}, PSNR: {psnr_score}")
                
            
            # Select best based on SSIM (priority) and PSNR
            if ssim_score > best_ssim or (ssim_score == best_ssim and psnr_score > best_psnr):
                print(f"Select best")
                best_ssim = ssim_score
                best_psnr = psnr_score
                best_org_img_edit = tmp_org_img_edit
                best_original_image_edit_without_background = tmp_original_image_edit_without_background
                
            if ssim_score > 0.95 and psnr_score > 20:
                break
            
            current_seed += 7
        
        org_img_edit = best_org_img_edit
        original_image_edit_without_background = best_original_image_edit_without_background
        print(f"Find best org_img_edit after {num_attempt} attempt")
        
        # Debug: Save best org_img_edit
        self._save_debug_image(org_img_edit, debug_index, task_id, "best_org_img_edit")
        debug_index += 1
        self._save_debug_image(original_image_edit_without_background, debug_index, task_id, "best_org_img_edit_rmbg")
        debug_index += 1

        # 1. Edit the image using Qwen Edit
        image_edited_1 = self.qwen_edit.edit_image(
            prompt_image=org_img_edit,
            seed=request.seed,
            prompt="Left three-quarters view. The object remains fixed in space. Preserve the original scale, proportions, camera distance, and color palette. Keep the original background unchanged. Lighting remains consistent with the scene but does not directly illuminate the object; no new light sources or directional lighting added. Maintain existing shadows as-is",
        )
        self._save_debug_image(image_edited_1, debug_index, task_id, "edited_left")
        debug_index += 1
        
        image_without_background_1 = self.rmbg.remove_background(image_edited_1.copy())
        
        

        trellis_result: Optional[TrellisResult] = None
        idx_best_result = 0
        
        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params

        # 3. Generate the 3D model
        s0 = time.time()
        trellis_result_1 = self.trellis.generate(
            TrellisRequest(
                images=[
                    original_image_without_background,
                    image_without_background_1,
                ],
                seed=request.seed,
                params=trellis_params,
            )
        )
        self.estimate_time_gen3d = time.time() - s0
        render_1 = render_image_combine(trellis_result_1.ply_file)
        
        # Debug: Save render_1
        self._save_debug_image(render_1, debug_index, task_id, "render_1_multi_view")
        debug_index += 1
        
        trellis_result = trellis_result_1
        best_render = render_1
        idx_best_result = 1
        
        if time.time() - t1 + self.estimate_time_gen3d + self.estimate_time_valid < 32:
            s0 = time.time()
            
            image_edited_2 = self.qwen_edit.edit_image(
                prompt_image=org_img_edit,
                seed=request.seed,
                prompt="Right three-quarters view. The object remains fixed in space. Preserve the original scale, proportions, camera distance, and color palette. Keep the original background unchanged. Lighting remains consistent with the scene but does not directly illuminate the object; no new light sources or directional lighting added. Maintain existing shadows as-is",
            )
            self._save_debug_image(image_edited_2, debug_index, task_id, "edited_right")
            debug_index += 1
            image_without_background_2 = self.rmbg.remove_background(image_edited_2.copy())
            
            trellis_result_2 = self.trellis.generate_single(
                TrellisRequest(
                    images=[
                        original_image_without_background,
                        image_without_background_2,
                    ],
                    seed=request.seed,
                    params=trellis_params,
                )
            )
            render_2 = render_image_combine(trellis_result_2.ply_file)
            
            # Debug: Save render_2
            self._save_debug_image(render_2, debug_index, task_id, "render_2_single_view")
            debug_index += 1
            
            self.estimate_time_gen3d = max(self.estimate_time_gen3d, time.time() - s0)
           
            s1 = time.time()
            # Convert image to RGB with white background for judging
            winner, avg_penalty_left, avg_penalty_right, issues = await judge_3d_duel(render_1, render_2, tmp_original_image_edit_with_white_bg)
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
                        original_image_without_background,
                    ],
                    seed=request.seed,
                    params=trellis_params,
                )
            )
            
            render_3 = render_image_combine(trellis_result_3.ply_file)
            
            # Debug: Save render_3
            self._save_debug_image(render_3, debug_index, task_id, "render_3_two_view")
            debug_index += 1
            
            self.estimate_time_gen3d = max(self.estimate_time_gen3d, time.time() - s0)

            s1 = time.time()
            # Convert image to RGB with white background for judging
            winner, avg_penalty_left, avg_penalty_right, issues = await judge_3d_duel(render_3, best_render, tmp_original_image_edit_with_white_bg)
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
        
        # Debug: Save final best render
        self._save_debug_image(best_render, debug_index, task_id, f"final_best_render_idx{idx_best_result}")
        debug_index += 1

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
        
        if self.settings.debug:
            logger.info(f"Debug: Saved {debug_index} images to {self.settings.output_dir / 'debug' / task_id}")
        
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
