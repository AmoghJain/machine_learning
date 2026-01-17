import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from image_dataset import ImageDataset

# data_paths
monet_path = "../../../data/gan-getting-started/monet_jpg"
image_path = "../../../data/gan-getting-started/photo_jpg"

# define constants
LR = 0.0002
EPOCHS = 10
BATCH_SIZE = 64
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = [
    transforms.Resize(int(IMAGE_SIZE*1.12), Image.BICUBIC),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

def main():
    dataloader = DataLoader(
        ImageDataset(monet_path, image_path, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    for epoch in range(EPOCHS):
        for batch_idx, data in enumerate(dataloader):
            real_monet = data["monet"].to(DEVICE)
            real_image = data["image"].to(DEVICE)

            print(real_monet.shape, real_image.shape)
        break

if __name__ == "__main__":
    main()