import torch
import torch.nn.functional as F


_TARGET_LAYERS = {
    "dec_last": lambda m: m.dec_blocks[-1].target.vmapped.conv,
    "enc_last": lambda m: m.enc_blocks[-1].target.vmapped.conv,
}


class SegGradCAM:
    """Seg-Grad-CAM for UniverSeg.

    Hooks an inner Conv2d inside a Vmap-wrapped ConvOp. Reduces the
    per-pixel segmentation logits to a single scalar (mean logit over
    predicted-positive pixels) before backprop.
    """

    def __init__(self, model, target_layer="dec_last"):
        if target_layer not in _TARGET_LAYERS:
            raise ValueError(
                f"Unknown target_layer={target_layer!r}; choices: {list(_TARGET_LAYERS)}"
            )
        self.model = model
        self.target_layer_name = target_layer
        self.target_layer = _TARGET_LAYERS[target_layer](model)

        self._activations = None
        self._gradients = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def close(self):
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __call__(self, query_img, support_imgs, support_masks):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.zero_grad()

        logits = self.model(query_img, support_imgs, support_masks)
        probs = torch.sigmoid(logits).detach()

        pos_mask = (probs > 0.5).float()
        if pos_mask.sum() < 1.0:
            k = max(1, int(0.05 * probs.numel()))
            flat = probs.view(-1)
            thresh = torch.topk(flat, k).values.min()
            pos_mask = (probs >= thresh).float()

        scalar = (logits * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)
        scalar.backward()

        activations = self._activations
        gradients = self._gradients
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire; check target layer path.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=query_img.shape[-2:], mode="bilinear", align_corners=False
        )

        cam_np = cam.detach().cpu().numpy()[0, 0]
        lo, hi = float(cam_np.min()), float(cam_np.max())
        if hi - lo > 1e-12:
            cam_np = (cam_np - lo) / (hi - lo)
        else:
            cam_np = cam_np * 0.0

        return cam_np, probs.detach().cpu().numpy()[0, 0]
