import torch
from PIL import Image
from torchvision import transforms

from ben2 import BEN_Base

from modules.background_removal.rmbg_manager import BackgroundRemovalService

class BEN2BackgroundRemovalService(BackgroundRemovalService):
    def _initialize_model_and_transforms(self) -> tuple[BEN_Base, transforms.Compose]:
        """
        Initialize BEN2 model and transforms.
        """
        model: BEN_Base | None = None

        transform = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size), 
                transforms.ToTensor(),
            ]
        )

        return model, transform
    
    def _load_model(self) -> BEN_Base:
        """
        Load the BEN2 background removal model.
        """
        model = BEN_Base.from_pretrained(self.settings.model_id)
        return model.to(self.device).eval()

    def _remove_background(self, image: Image) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Remove the background from the image.
        """
        rgb_image = image.convert('RGB')
        rgb_tensor = self.transforms(rgb_image).to(self.device)


        with torch.no_grad():
            rgba = self.model.inference(rgb_image.copy())

        rgba = rgba.convert("RGBA")
        alpha = rgba.split()[-1]  
        mask = self.transforms(alpha).to(self.device)
        mask = mask.squeeze(0).mul_(255).int().div(255).float()
        mask = mask.clamp(0, 1)
        return rgb_tensor, mask