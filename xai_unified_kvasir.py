import random
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from universeg import universeg
from universeg.xai import (
    MedicalFewShotDataset,
    SegGradCAM,
    run_shap_gradient,
    save_combined_xai_panel,
    set_seed,
)

BACKGROUND_SIZE = 8

IMAGE_DIR = "data/kvasir/images"
MASK_DIR = "data/kvasir/masks"


def sample_background_queries(dataset, n, device):
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    imgs = [dataset._load_item(i)[0] for i in idxs]
    return torch.stack(imgs).to(device).detach()


def main():
    TARGET_SEED = None  # set to an int (e.g. 1718345) to reproduce a past run

    current_seed = TARGET_SEED if TARGET_SEED is not None else int(time.time())
    print(f"\n=== USING SEED: {current_seed} ===\n")

    set_seed(current_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = MedicalFewShotDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        dataset_type="kvasir",
        support_size=3,
        transform=transform,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("Model initializing...")
    model = universeg(pretrained=True).to(device)
    model.eval()

    support_imgs, support_masks, query_img, query_mask, query_name = next(iter(loader))
    image_name = query_name[0]
    print(f"\n---> PROCESSING KVASIR QUERY: {image_name} <---")

    query_img = query_img.to(device).detach().clone()
    support_imgs = support_imgs.to(device).detach().clone()
    support_masks = support_masks.to(device).detach().clone()
    query_mask_np = query_mask.cpu().squeeze().numpy()
    query_np = query_img.cpu().squeeze().numpy()

    print(f"Sampling {BACKGROUND_SIZE} background queries (frozen before inference)...")
    background_queries = sample_background_queries(dataset, BACKGROUND_SIZE, device)

    print("Step A: base forward pass...")
    with torch.no_grad():
        base_logits = model(query_img, support_imgs, support_masks)
        base_pred_tensor = torch.sigmoid(base_logits).detach()
        base_pred = base_pred_tensor.cpu().squeeze().numpy()
        frozen_base_pred_mask = (base_pred_tensor > 0.5).float()

    print("Step B: Grad-CAM (dec_last)...")
    with SegGradCAM(model, target_layer="dec_last") as gradcam_fn:
        gradcam_map, _ = gradcam_fn(query_img, support_imgs, support_masks)
    model.eval()

    print("Step C: SHAP GradientExplainer...")
    t0 = time.time()
    shap_map = run_shap_gradient(
        model=model,
        query_img=query_img,
        support_imgs=support_imgs,
        support_masks=support_masks,
        background_queries=background_queries,
        frozen_base_pred_mask=frozen_base_pred_mask,
        device=device,
    )
    print(
        f"SHAP done in {time.time() - t0:.1f}s; "
        f"range: [{shap_map.min():.3e}, {shap_map.max():.3e}]"
    )

    stem = image_name.rsplit(".", 1)[0]
    out_path = f"xai_panel_kvasir_{stem}_seed_{current_seed}.png"
    save_combined_xai_panel(
        query=query_np,
        gt=query_mask_np,
        pred=base_pred,
        gradcam=gradcam_map,
        shap_map=shap_map,
        out_path=out_path,
        title=image_name,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
