import numpy as np
import torch


def run_shap_gradient(model, query_img, support_imgs, support_masks, background_queries, device):
    """Run shap.GradientExplainer on the UniverSeg model.

    The model is wrapped so SHAP sees a single input (the query image) and a
    single scalar output (mean foreground probability). Supports are held
    fixed inside the closure so the explanation isolates query-image
    attributions.

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
    for p in model.parameters():
        p.requires_grad_(True)

    fixed_sup_imgs = support_imgs.detach()
    fixed_sup_masks = support_masks.detach()

    class Wrapped(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, q):
            n = q.shape[0]
            sup_i = fixed_sup_imgs.expand(n, -1, -1, -1, -1)
            sup_l = fixed_sup_masks.expand(n, -1, -1, -1, -1)
            logits = self.base(q, sup_i, sup_l)
            probs = torch.sigmoid(logits)
            return probs.mean(dim=(1, 2, 3), keepdim=True).view(n, 1)

    wrapped = Wrapped(model).to(device)
    wrapped.eval()

    background_queries = background_queries.to(device)
    explainer = shap.GradientExplainer(wrapped, background_queries)

    query = query_img.to(device)
    shap_values = explainer.shap_values(query)

    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    sv = np.asarray(sv)
    while sv.ndim > 2:
        sv = sv.squeeze(0) if sv.shape[0] == 1 else sv.sum(axis=0)

    return sv
