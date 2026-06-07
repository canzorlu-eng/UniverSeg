import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

from universeg import universeg
from universeg.xai import MedicalFewShotDataset

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

    # NOTE: Now loader returns 5 items (query_name added)
    support_imgs, support_masks, query_img, query_mask, query_name = next(iter(loader))
    
    # query_name is a tuple, extract the string inside
    image_name = query_name[0]
    print(f"\n---> PROCESSING QUERY IMAGE: {image_name} <---")
    
    support_imgs = support_imgs.to(device)
    support_masks = support_masks.to(device)
    query_img = query_img.to(device)

    print("Making prediction...")
    with torch.no_grad():
        pred_mask = model(query_img, support_imgs, support_masks)
        pred_mask = torch.sigmoid(pred_mask)

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Query: {image_name}", fontsize=10)
    plt.imshow(query_img.cpu().squeeze(), cmap="gray")
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(query_mask.cpu().squeeze(), cmap="gray")
    
    plt.subplot(1, 3, 3)
    plt.title("Model Prediction")
    plt.imshow(pred_mask.cpu().squeeze(), cmap="gray")
    
    plt.tight_layout()
    plt.savefig("ilk_deneme_sonucu.png")
    print(f"Success! Output saved as 'ilk_deneme_sonucu.png'.")

if __name__ == "__main__":
    main()