import torch
import sys
from datetime import datetime
import numpy as np
import random


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    # Define the indices for the lower triangular part, including the diagonal
    tril_indices = torch.tril_indices(row=3, col=3, offset=0, device=L.device)

    # Extract the lower triangular elements from each matrix in the batch
    lower_triangular_elements = L[:, tril_indices[0], tril_indices[1]]

    # Select the specific elements corresponding to the desired positions
    uncertainty = lower_triangular_elements[:, [0, 1, 2, 4, 5, 8]]

    return uncertainty


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    return strip_lowerdiag(sym)


def build_rotation(quaternions: torch.Tensor) -> torch.Tensor:
    # Normalize the quaternions
    quaternions = torch.nn.functional.normalize(quaternions, p=2, dim=1)

    # Extract individual components
    r, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Compute the rotation matrices
    rotation_matrices = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1).reshape(-1, 3, 3)

    return rotation_matrices

def build_scaling_rotation(scaling_mat: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    batch_size = scaling_mat.shape[0]

    # Initialize scaling matrices L as identity matrices
    L = torch.eye(3, device=scaling_mat.device).unsqueeze(0).repeat(batch_size, 1, 1)

    # Set the diagonal elements to the scaling factors
    L *= scaling_mat.unsqueeze(2)

    # Compute rotation matrices R using the provided build_rotation function
    R = build_rotation(quaternions)

    # Perform batched matrix multiplication
    L = torch.bmm(R, L)

    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
