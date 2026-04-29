import torch
import numpy as np


def simplest_color_balance(im_org, num=255):
    """
    Args:
        im_org (torch.Tensor or numpy.ndarray): Input image.
            - For color: shape (H, W, 3) or (3, H, W) - will auto-detect.
            - For grayscale: shape (H, W) or (1, H, W).
            Expected dtype: uint8 (0-255) or float (will be converted to float for processing).
        num (int): Target maximum value after stretching (default 255, for uint8 output).
    Returns:
        torch.Tensor or numpy.ndarray: Color balanced image with same shape and type as input.
    """
    input_is_numpy = isinstance(im_org, np.ndarray)
    if input_is_numpy:
        orig_dtype = im_org.dtype
        im_tensor = torch.from_numpy(im_org).float()
    elif isinstance(im_org, torch.Tensor):
        orig_dtype = im_org.dtype
        im_tensor = im_org.float()
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")

    orig_shape = im_tensor.shape
    if len(orig_shape) == 3:
        if orig_shape[-1] in (1, 3):
            # (H, W, C) layout
            h, w, c = orig_shape
            channel_dim = -1
            im_tensor = im_tensor.permute(2, 0, 1)  # (C, H, W)
        elif orig_shape[0] in (1, 3):
            # (C, H, W) layout
            c, h, w = orig_shape
            channel_dim = 0
        else:
            raise ValueError("Unable to determine channel dimension for 3D tensor")
    elif len(orig_shape) == 2:
        # Grayscale (H, W) -> treat as (1, H, W)
        h, w = orig_shape
        c = 1
        im_tensor = im_tensor.view(1, h, w)
        channel_dim = 0
    else:
        raise ValueError("Input must be 2D (grayscale) or 3D (color) tensor")

    N = h * w
    im_flat = im_tensor.view(c, N)  # (C, N)

    if c == 3:
        sums = im_flat.sum(dim=1)  # (C,)
        max_sum = sums.max()
        ratio = max_sum / sums
        satLevel1 = 0.005 * ratio
        satLevel2 = 0.005 * ratio
    else:
        satLevel1 = torch.tensor([0.001], device=im_tensor.device)
        satLevel2 = torch.tensor([0.005], device=im_tensor.device)

    im_balanced_flat = torch.empty_like(im_flat)
    for ch in range(c):
        q_low = satLevel1[ch] if satLevel1[ch] <= 1.0 else 1.0
        q_high = 1 - satLevel2[ch] if (1 - satLevel2[ch]) >= 0.0 else 0.0
        if q_low >= q_high:
            low_val = im_flat[ch].min()
            high_val = im_flat[ch].max()
        else:
            q = torch.tensor([q_low, q_high], device=im_tensor.device)
            low_val, high_val = torch.quantile(im_flat[ch], q)

        clipped = im_flat[ch].clamp(low_val, high_val)
        min_val = clipped.min()
        max_val = clipped.max()
        if max_val > min_val:
            stretched = (clipped - min_val) * num / (max_val - min_val)
        else:
            stretched = clipped
        im_balanced_flat[ch] = stretched

    im_balanced = im_balanced_flat.view(c, h, w)

    if c == 1 and len(orig_shape) == 2:
        out_tensor = im_balanced.view(h, w)
    else:
        if channel_dim == -1:
            out_tensor = im_balanced.permute(1, 2, 0)
        else:
            out_tensor = im_balanced

    out_tensor = out_tensor.clamp(0, num).byte()

    if input_is_numpy:
        return out_tensor.cpu().numpy()
    else:
        return out_tensor