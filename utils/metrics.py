import numpy as np
from skimage.metrics import structural_similarity as ssim_func

def calculate_aec(message, img_shape):
    # AEC (bpp) = Tổng số bit / Tổng số pixel
    total_bits = len(message) * 8 + 8
    return total_bits / (img_shape[0] * img_shape[1])

def calculate_psnr(img_c, img_s):
    mse = np.mean((img_c.astype(np.float64) - img_s.astype(np.float64)) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img_c, img_s):
    # win_size=3 để tránh lỗi với ảnh nhỏ
    return ssim_func(img_c, img_s, win_size=3)

def calculate_uiqi(img_c, img_s):
    u, v = img_c.astype(np.float64), img_s.astype(np.float64)
    mu_u, mu_v = np.mean(u), np.mean(v)
    var_u, var_v = np.var(u), np.var(v)
    cov_uv = np.mean((u - mu_u) * (v - mu_v))
    denom = (var_u + var_v) * (mu_u**2 + mu_v**2)
    if denom == 0: return 1.0
    return (4 * cov_uv * mu_u * mu_v) / denom

def calculate_ncc(img_c, img_s):
    c, s = img_c.astype(np.float64), img_s.astype(np.float64)
    num = np.sum(c * s)
    den = np.sqrt(np.sum(c**2) * np.sum(s**2))
    return num / den if den != 0 else 0