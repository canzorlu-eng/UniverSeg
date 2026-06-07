import numpy as np
import torch


def _as_2d_shap_map(shap_values, spatial_shape):
    """Extract SHAP attributions as a strict (H, W) map aligned to the query grid."""
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    sv = np.asarray(shap_values, dtype=np.float32)
    h, w = spatial_shape

    # GradientExplainer returns attributions with the same layout as model input:
    # (batch, channels, H, W). Index explicitly — never sum axes that would
    # collapse spatial structure or pool values into a corner artifact.
    if sv.ndim == 4 and sv.shape[-2:] == (h, w):
        return np.squeeze(sv[0, 0])

    sv = np.squeeze(sv)
    if sv.ndim == 2 and sv.shape == (h, w):
        return sv

    if sv.ndim == 3 and sv.shape[-2:] == (h, w) and sv.shape[0] == 1:
        return sv[0]

    raise ValueError(
        f"Expected SHAP attributions aligned to ({h}, {w}), got shape {sv.shape}."
    )


def run_shap_gradient(
    model,
    query_img,
    support_imgs,
    support_masks,
    background_queries,
    frozen_base_pred_mask,
    device,
):
    """Run shap.GradientExplainer on the UniverSeg model.

    Supports and the base-run binary lesion mask are frozen inside a wrapper so
    SHAP only perturbs the query image. The scalar target is the summed
    predicted foreground probability over lesion pixels (not raw logit mass).

    Args:
        model: trained UniverSeg model (on `device`, eval mode).
        query_img: (1, 1, H, W) tensor.
        support_imgs: (1, S, 1, H, W) tensor.
        support_masks: (1, S, 1, H, W) tensor.
        background_queries: (N, 1, H, W) tensor of background query images.
        frozen_base_pred_mask: (1, 1, H, W) binary mask from the base forward pass.
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

    pred_mask = frozen_base_pred_mask.to(device).float().detach()
    if pred_mask.dim() == 2:
        pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)
    elif pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(0)

    if query.shape[1:] != background.shape[1:]:
        raise ValueError(
            f"query_img spatial shape {tuple(query.shape[1:])} must match "
            f"background_queries {tuple(background.shape[1:])}."
        )
    if pred_mask.shape[-2:] != query.shape[-2:]:
        raise ValueError(
            f"frozen_base_pred_mask spatial shape {tuple(pred_mask.shape[-2:])} must "
            f"match query_img {tuple(query.shape[-2:])}."
        )

    spatial_shape = query.shape[-2:]

    class QueryOnlyWrapper(torch.nn.Module):
        """Expose only the query image to SHAP; freeze supports and lesion mask."""

        def __init__(self, base, frozen_support_imgs, frozen_support_masks, lesion_mask):
            super().__init__()
            self.base = base
            self.register_buffer("support_imgs", frozen_support_imgs.detach())
            self.register_buffer("support_masks", frozen_support_masks.detach())
            self.register_buffer("pred_mask", lesion_mask.detach())

        def forward(self, query_batch):
            batch_size = query_batch.shape[0]
            sup_imgs = self.support_imgs.expand(batch_size, -1, -1, -1, -1)
            sup_masks = self.support_masks.expand(batch_size, -1, -1, -1, -1)
            logits = self.base(query_batch, sup_imgs, sup_masks)
            probs = torch.sigmoid(logits)
            mask = self.pred_mask.expand(batch_size, -1, -1, -1)
            return (probs * mask).sum(dim=(1, 2, 3), keepdim=True)

    wrapped = QueryOnlyWrapper(
        model,
        support_imgs.to(device).float(),
        support_masks.to(device).float(),
        pred_mask,
    ).to(device)
    wrapped.eval()

    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = explainer.shap_values(query)

    shap_map = _as_2d_shap_map(shap_values, spatial_shape)

    for param in model.parameters():
        param.requires_grad_(False)

    return shap_map
