import random
import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from universeg import universeg
from universeg.xai import MedicalFewShotDataset, run_shap_gradient, save_panel


BACKGROUND_SIZE = 8


def sample_background(dataset, n, device):
    random.seed(0)
    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    imgs = []
    for i in idxs:
        img, _, _ = dataset._load_item(i)
        imgs.append(img)
    return torch.stack(imgs).to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = MedicalFewShotDataset(
        image_dir="data/isic/images",
        mask_dir="data/isic/masks",
        dataset_type="isic",
        support_size=3,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Model initializing...")
    model = universeg(pretrained=True).to(device)
    model.eval()

    support_imgs, support_masks, query_img, query_mask, query_name = next(iter(loader))
    image_name = query_name[0]
    print(f"\n---> PROCESSING QUERY IMAGE: {image_name} <---")

    support_imgs = support_imgs.to(device)
    support_masks = support_masks.to(device)
    query_img = query_img.to(device)

    print(f"Sampling {BACKGROUND_SIZE} background queries...")
    background = sample_background(dataset, BACKGROUND_SIZE, device)

    with torch.no_grad():
        pred_logits = model(query_img, support_imgs, support_masks)
        pred_prob = torch.sigmoid(pred_logits).cpu().squeeze().numpy()

    print("Running SHAP GradientExplainer...")
    t0 = time.time()
    shap_map = run_shap_gradient(
        model=model,
        query_img=query_img,
        support_imgs=support_imgs,
        support_masks=support_masks,
        background_queries=background,
        device=device,
    )
    print(f"SHAP done in {time.time() - t0:.1f}s; shap range: [{shap_map.min():.3e}, {shap_map.max():.3e}]")

    out_path = f"xai_shap_{image_name.split('.')[0]}.png"
    save_panel(
        query=query_img.cpu().squeeze().numpy(),
        gt=query_mask.cpu().squeeze().numpy(),
        pred=pred_prob,
        heatmap=shap_map,
        out_path=out_path,
        title=image_name,
        heatmap_label="SHAP (signed)",
        heatmap_cmap="coolwarm",
        symmetric=True,
        overlay=False,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
