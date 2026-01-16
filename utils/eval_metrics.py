# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity

# def calculate_joint_metric(sr, hr):
#     """
#     sr, hr: torch tensors in range [0,1], shape (3,H,W)
#     """
#     sr = sr.permute(1,2,0).cpu().numpy() * 255
#     hr = hr.permute(1,2,0).cpu().numpy() * 255

#     sr = sr.astype(np.uint8)
#     hr = hr.astype(np.uint8)

#     psnr = peak_signal_noise_ratio(hr, sr, data_range=255)
#     ssim, _ = structural_similarity(hr, sr, full=True, multichannel=True, channel_axis=-1)

#     joint = 40 * ssim + psnr
#     return joint, psnr, ssim

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def calculate_joint_metric(sr, hr):
    """
    sr, hr: torch tensors (C, H, W) in [0,1]
    """
    sr = sr.detach().cpu().numpy().transpose(1, 2, 0)
    hr = hr.detach().cpu().numpy().transpose(1, 2, 0)

    sr = np.clip(sr, 0, 1)
    hr = np.clip(hr, 0, 1)

    psnr_val = peak_signal_noise_ratio(
        hr, sr, data_range=1.0
    )

    ssim_val = structural_similarity(
        hr,
        sr,
        data_range=1.0,
        channel_axis=2
    )

    joint = psnr_val + 40.0 * ssim_val
    return joint, psnr_val, ssim_val
