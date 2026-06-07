import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from universeg import universeg

class MedicalFewShotDataset(Dataset):
    def __init__(self, image_dir, mask_dir, dataset_type="isic", support_size=3, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type.lower()
        self.support_size = support_size
        self.transform = transform
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
        
    def __len__(self):
        return len(self.image_files)

    def _load_item(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        if self.dataset_type == "isic":
            mask_name = img_name.replace(".jpg", "_segmentation.png")
            mask_path = os.path.join(self.mask_dir, mask_name)
        elif self.dataset_type == "kvasir":
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        else:
            raise ValueError("Not supported dataset type!")
        
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0.5).float()
            
        # DİKKAT: Artık dosya adını (img_name) da döndürüyoruz
        return image, mask, img_name

    def __getitem__(self, idx):
        # Query (Sorgu) resmi ve adını al
        query_img, query_mask, query_name = self._load_item(idx)
        
        available_indices = list(range(len(self.image_files)))
        available_indices.remove(idx)
        
        actual_support_size = min(self.support_size, len(available_indices))
        support_indices = random.sample(available_indices, actual_support_size)
        
        support_imgs, support_masks = [], []
        for s_idx in support_indices:
            # Support resimlerinin isimlerine ihtiyacımız yok, o yüzden "_" ile yoksayıyoruz
            s_img, s_mask, _ = self._load_item(s_idx)
            support_imgs.append(s_img)
            support_masks.append(s_mask)
            
        # DİKKAT: return kısmına query_name'i ekledik
        return torch.stack(support_imgs), torch.stack(support_masks), query_img, query_mask, query_name

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

    print("Model Initializing...")
    model = universeg(pretrained=True).to(device)
    model.eval()

    # DİKKAT: Artık loader'dan 5 parça veri çekiyoruz (query_name eklendi)
    support_imgs, support_masks, query_img, query_mask, query_name = next(iter(loader))
    
    # query_name bir tuple olarak gelir, içindeki string'i alıyoruz
    resim_adi = query_name[0]
    print(f"\n---> İŞLENEN RESİM (QUERY): {resim_adi} <---")
    
    support_imgs = support_imgs.to(device)
    support_masks = support_masks.to(device)
    query_img = query_img.to(device)

    print("Tahmin yapılıyor...")
    with torch.no_grad():
        pred_mask = model(query_img, support_imgs, support_masks)
        pred_mask = torch.sigmoid(pred_mask)

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Query: {resim_adi}", fontsize=10)
    plt.imshow(query_img.cpu().squeeze(), cmap="gray")
    
    plt.subplot(1, 3, 2)
    plt.title("Gerçek Maske")
    plt.imshow(query_mask.cpu().squeeze(), cmap="gray")
    
    plt.subplot(1, 3, 3)
    plt.title("Model Tahmini")
    plt.imshow(pred_mask.cpu().squeeze(), cmap="gray")
    
    plt.tight_layout()
    plt.savefig("ilk_deneme_sonucu.png")
    print(f"Tebrikler! Çıktı 'ilk_deneme_sonucu.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()