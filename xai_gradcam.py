import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from universeg import universeg
from universeg.xai import MedicalFewShotDataset, SegGradCAM, save_panel


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

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("Model initializing...")
    model = universeg(pretrained=True).to(device)
    model.eval()

    support_imgs, support_masks, query_img, query_mask, query_name = next(iter(loader))
    image_name = query_name[0]
    print(f"\n---> PROCESSING QUERY IMAGE: {image_name} <---")

    support_imgs = support_imgs.to(device)
    support_masks = support_masks.to(device)
    query_img = query_img.to(device)

    print("Computing Seg-Grad-CAM (target layer: dec_last)...")
    with SegGradCAM(model, target_layer="dec_last") as cam_method:
        cam, pred_prob = cam_method(query_img, support_imgs, support_masks)

    print(f"CAM stats: min={cam.min():.4f} max={cam.max():.4f} mean={cam.mean():.4f} std={cam.std():.4f}")
    print(f"Predicted foreground pixels (>0.5): {int((pred_prob > 0.5).sum())}")

    out_path = f"xai_gradcam_{image_name.split('.')[0]}.png"
    save_panel(
        query=query_img.cpu().squeeze().numpy(),
        gt=query_mask.cpu().squeeze().numpy(),
        pred=pred_prob,
        heatmap=cam,
        out_path=out_path,
        title=image_name,
        heatmap_label="Grad-CAM (dec_last)",
        heatmap_cmap="jet",
        symmetric=False,
        overlay=True,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
