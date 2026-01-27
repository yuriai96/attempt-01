from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation

from logger_config import logger
from modules.background_removal.birefnet import BiRefNet
from modules.background_removal.birefnet.utils import check_state_dict
from modules.background_removal.rmbg_manager import BackgroundRemovalService

class KryvielBackgroundRemovalService(BackgroundRemovalService):
    def _initialize_model_and_transforms(self) -> tuple[BiRefNet, transforms.Compose]:
        """
        Initialize Kryviel model and transforms.
        """
        model: BiRefNet | None = None

        transform = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size),
                transforms.ToTensor(),
            ]
        )

        self.mask_threshold = self.settings.background_removal_mask_threshold
        self.biref_use_fp16 = bool(self.settings.birefnet_use_fp16 and torch.cuda.is_available())
        self.biref_transforms = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.hf_model = None

        return model, transform
    
    def _load_model(self) -> BiRefNet:
        """
        Load the Kryviel background removal model.
        """
        model = self._load_birefnet()
        self.hf_model = self._load_hf_birefnet()
        return model

    def _remove_background(self, image: Image.Image, threshold: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Remove the background from the image.
        """
        self.ensure_ready()
        mask_threshold = self.mask_threshold if threshold is None else threshold

        rgb_image = image.convert("RGB")
        rgb_tensor = self.transforms(rgb_image).to(self.device)

        with torch.no_grad():
            mask_local = self._predict_biref_mask(rgb_image)

        mask_local_bin = (mask_local > mask_threshold).float()
        chosen_mask = mask_local_bin

        if self.hf_model is not None:
            mask_hf = self._predict_hf_mask(rgb_image)
            if mask_hf is not None:
                mask_hf_bin = (mask_hf > mask_threshold).float()
                local_area = float(mask_local_bin.sum())
                hf_area = float(mask_hf_bin.sum())
                if hf_area > local_area:
                    chosen_mask = mask_hf_bin
                    logger.info("Selected HF BiRefNet mask (less background removed).")
                else:
                    logger.info("Selected local BiRefNet mask (less background removed).")

        chosen_area = float(chosen_mask.sum())
        if chosen_area == 0:
            logger.warning("BiRefNet mask is empty; output may be fully transparent.")

        return rgb_tensor, chosen_mask

    def _predict_biref_mask(self, image: Image.Image) -> torch.Tensor:
        """
        Predict foreground mask with BiRefNet.
        """
        input_tensor = self.biref_transforms(image).unsqueeze(0).to(self.device)
        if self.biref_use_fp16:
            input_tensor = input_tensor.half()

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid()

        pred = preds[0].squeeze(0).float()
        return pred

    def _predict_hf_mask(self, image: Image.Image) -> torch.Tensor | None:
        """
        Predict foreground mask with HuggingFace BiRefNet.
        """
        if self.hf_model is None:
            return None

        input_tensor = self.biref_transforms(image).unsqueeze(0).to(self.device)
        if self.biref_use_fp16:
            input_tensor = input_tensor.half()

        with torch.no_grad():
            preds = self.hf_model(input_tensor)[-1].sigmoid()

        pred = preds[0].squeeze(0).float()
        return pred

    def _load_birefnet(self) -> BiRefNet:
        """
        Load custom BiRefNet weights from HuggingFace Hub.
        """
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        
        logger.info(
            f"Loading custom BiRefNet weights from HF: "
            f"{self.settings.birefnet_hf_repo}/{self.settings.birefnet_hf_filename}"
        )
        try:
            ckpt_path = hf_hub_download(
                repo_id=self.settings.birefnet_hf_repo,
                filename=self.settings.birefnet_hf_filename,
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"BiRefNet HF download failed: {exc}"
            ) from exc

        model = BiRefNet(bb_pretrained=False)
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        if self.biref_use_fp16:
            model.half()
        logger.info("Custom BiRefNet weights loaded.")
        return model

    def _load_hf_birefnet(self) -> AutoModelForImageSegmentation | None:
        """
        Load BiRefNet from HuggingFace (previous method).
        """
        try:
            logger.info(f"Loading reference HF BiRefNet model: {self.settings.model_id}")
            model = AutoModelForImageSegmentation.from_pretrained(
                self.settings.model_id, trust_remote_code=True
            )
            model = model.to(self.device)
            model.eval()
            if self.biref_use_fp16:
                model.half()
            return model
        except Exception as exc:
            logger.error(f"Failed to load HF BiRefNet model: {exc}")
            return None
