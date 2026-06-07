import numpy as np
import torch


def _as_2d_shap_map(shap_values, spatial_shape):
    """Reduce GradientExplainer output to a strict (H, W) attribution map."""
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    sv = np.asarray(shap_values, dtype=np.float32)
    h, w = spatial_shape

    # Remove singleton batch / channel dims first (preferred path).
    sv = np.squeeze(sv)
    if sv.ndim == 2 and sv.shape == (h, w):
        return sv

    # Fallback layouts SHAP occasionally returns for image models.
    if sv.ndim == 3:
        if sv.shape[-2:] == (h, w):
            return sv[0] if sv.shape[0] == 1 else sv.sum(axis=0)
        if sv.shape[:2] == (h, w):
            return sv[..., 0] if sv.shape[-1] == 1 else sv.sum(axis=-1)

    if sv.ndim == 4 and sv.shape[-2:] == (h, w):
        sv = sv[0]
        return sv[0] if sv.shape[0] == 1 else sv.sum(axis=0)

    raise ValueError(
        f"Expected SHAP attributions reducible to ({h}, {w}), got shape {sv.shape}."
    )


def run_shap_gradient(model, query_img, support_imgs, support_masks, background_queries, device):
    """Run shap.GradientExplainer on the UniverSeg model.

    Supports are frozen inside a wrapper so SHAP only perturbs the query image.
    Segmentation logits are summed to a scalar so GradientExplainer can backprop.

    Args:
        model: trained UniverSeg model (on `device`, eval mode).
        query_img: (1, 1, H, W) tensor.
        support_imgs: (1, S, 1, H, W) tensor.
        support_masks: (1, S, 1, H, W) tensor.
        background_queries: (N, 1, H, W) tensor of background query images.
        device: torch device.

    Returns:
        shap_map: (H, W) numpy array (signed) — SHAP values for the query.
    """
    import shap

    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)

    query = query_img.to(device).float()
    if query.dim() == 3:
        query = query.unsqueeze(0)

    background = background_queries.to(device).float().detach()
    if background.dim() == 3:
        background = background.unsqueeze(0)

    if query.shape[1:] != background.shape[1:]:
        raise ValueError(
            f"query_img spatial shape {tuple(query.shape[1:])} must match "
            f"background_queries {tuple(background.shape[1:])}."
        )

    spatial_shape = query.shape[-2:]

    class QueryOnlyWrapper(torch.nn.Module):
        """Expose only the query image to SHAP; freeze support set."""

        def __init__(self, base, frozen_support_imgs, frozen_support_masks):
            super().__init__()
            self.base = base
            self.register_buffer("support_imgs", frozen_support_imgs.detach())
            self.register_buffer("support_masks", frozen_support_masks.detach())

        def forward(self, query_batch):
            batch_size = query_batch.shape[0]
            sup_imgs = self.support_imgs.expand(batch_size, -1, -1, -1, -1)
            sup_masks = self.support_masks.expand(batch_size, -1, -1, -1, -1)
            logits = self.base(query_batch, sup_imgs, sup_masks)
            # Scalar per sample: total predicted foreground logit mass.
            return logits.sum(dim=(1, 2, 3), keepdim=True)

    wrapped = QueryOnlyWrapper(
        model,
        support_imgs.to(device).float(),
        support_masks.to(device).float(),
    ).to(device)
    wrapped.eval()

    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = explainer.shap_values(query)

    shap_map = _as_2d_shap_map(shap_values, spatial_shape)

    for param in model.parameters():
        param.requires_grad_(False)

    return shap_map
