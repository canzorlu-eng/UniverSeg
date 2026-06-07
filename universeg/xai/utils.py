import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm


def set_seed(seed=42):
    """Lock RNG state across Python, NumPy, and PyTorch (CPU/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def overlay_heatmap(img_2d, heat_2d, alpha=0.5, cmap="jet"):
    img = normalize01(img_2d)
    heat = normalize01(heat_2d)
    img_rgb = np.stack([img, img, img], axis=-1)
    colormap = cm.get_cmap(cmap)
    heat_rgb = colormap(heat)[..., :3]
    blended = (1.0 - alpha) * img_rgb + alpha * heat_rgb
    return np.clip(blended, 0.0, 1.0)


def save_panel(query, gt, pred, heatmap, out_path, title,
               heatmap_label="XAI Heatmap", heatmap_cmap="jet", symmetric=False,
               overlay=True):
    fig = plt.figure(figsize=(14, 4))

    ax1 = plt.subplot(1, 4, 1)
    ax1.set_title(f"Query: {title}", fontsize=10)
    ax1.imshow(query, cmap="gray")

    ax2 = plt.subplot(1, 4, 2)
    ax2.set_title("Ground Truth Mask")
    ax2.imshow(gt, cmap="gray")

    ax3 = plt.subplot(1, 4, 3)
    ax3.set_title("Model Prediction")
    ax3.imshow(pred, cmap="gray")

    ax4 = plt.subplot(1, 4, 4)
    ax4.set_title(heatmap_label)
    if symmetric:
        vmax = float(np.abs(heatmap).max()) or 1e-6
        im = ax4.imshow(heatmap, cmap=heatmap_cmap, vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    elif overlay:
        ax4.imshow(overlay_heatmap(query, heatmap, alpha=0.5, cmap=heatmap_cmap))
    else:
        im = ax4.imshow(heatmap, cmap=heatmap_cmap)
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def save_combined_xai_panel(
    query,
    gt,
    pred,
    gradcam,
    shap_map,
    out_path,
    title,
    gradcam_cmap="jet",
    shap_cmap="coolwarm",
):
    """Save a 1x5 panel: query, GT, prediction, Grad-CAM overlay, SHAP heatmap."""
    fig = plt.figure(figsize=(18, 4))

    ax1 = plt.subplot(1, 5, 1)
    ax1.set_title(f"Query: {title}", fontsize=10)
    ax1.imshow(query, cmap="gray")
    ax1.axis("off")

    ax2 = plt.subplot(1, 5, 2)
    ax2.set_title("Ground Truth Mask")
    ax2.imshow(gt, cmap="gray")
    ax2.axis("off")

    ax3 = plt.subplot(1, 5, 3)
    ax3.set_title("Base Prediction Mask")
    ax3.imshow(pred, cmap="gray")
    ax3.axis("off")

    ax4 = plt.subplot(1, 5, 4)
    ax4.set_title("Grad-CAM Overlay")
    ax4.imshow(overlay_heatmap(query, gradcam, alpha=0.5, cmap=gradcam_cmap))
    ax4.axis("off")

    ax5 = plt.subplot(1, 5, 5)
    ax5.set_title("SHAP Heatmap")
    vmax = float(np.abs(shap_map).max()) or 1e-6
    im = ax5.imshow(shap_map, cmap=shap_cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
    ax5.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
