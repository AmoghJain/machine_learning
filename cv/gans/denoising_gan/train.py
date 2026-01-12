import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from generator import Generator
from discriminator import Discriminator

# define parameters
LR = 0.0001
EPOCHS = 40
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# add noise
def add_gaussian_noise(image_tensor, noise_factor=0.5):
    noise_tensor = torch.randn_like(image_tensor) * noise_factor
    noisy_image = image_tensor + noise_tensor
    return noisy_image

# load data
print("Loading data...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.1307), (.3801))
])

train_dataset = MNIST("../../../data", download=False, transform=transform, train=True)
test_dataset = MNIST("../../../data", download=False, transform=transform, train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# instantiate networks
disc = Discriminator().to(DEVICE)
gen = Generator().to(DEVICE)

# define loss functions and optimizers
disc_opt = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
gen_opt = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))

gan_loss = nn.BCEWithLogitsLoss()
pixel_loss = nn.L1Loss()

# train model
print("Training started...")

for epoch in range(EPOCHS):
    for batch_idx, (clean_images, _) in enumerate(train_loader):
        clean_images = clean_images.to(DEVICE)
        
        ## train discriminator
        disc_opt.zero_grad()

        # prepare data
        noisy_images = add_gaussian_noise(clean_images)
        denoised_images = gen(noisy_images)

        # create labels
        clean_labels = torch.ones(clean_images.size(0), 1).to(DEVICE)
        noisy_labels = torch.zeros(noisy_images.size(0), 1).to(DEVICE)

        # discriminator on clean images
        disc_clean_outputs = disc(clean_images)
        disc_clean_loss = gan_loss(disc_clean_outputs, clean_labels)

        # discriminator on noisy images
        disc_noisy_outputs = disc(denoised_images.detach())
        disc_noisy_loss = gan_loss(disc_noisy_outputs, noisy_labels)

        disc_loss = (disc_clean_loss + disc_noisy_loss)/2.
        disc_loss.backward()
        disc_opt.step()

        ## train generator
        gen_opt.zero_grad()

        disc_output_for_gen = disc(denoised_images)
        gen_gan_loss = gan_loss(disc_output_for_gen, clean_labels)

        gen_pixel_loss = pixel_loss(denoised_images, clean_images)

        gen_loss = gen_gan_loss + 100*gen_pixel_loss
        gen_loss.backward()
        gen_opt.step()
    print(f"Epoch: {epoch+1}/{EPOCHS}, Gen loss: {gen_loss.item()}, Disc loss: {disc_loss.item()}")

print("Training finished...")

gen.eval()
with torch.no_grad():
    sample_clean, _ = next(iter(test_loader))
    sample_clean = sample_clean.to(DEVICE)[:8] # Take 8 images
    sample_noisy = add_gaussian_noise(sample_clean).to(DEVICE)
    sample_denoised = gen(sample_noisy)

    # Plot
    fig, axs = plt.subplots(3, 8, figsize=(12, 8))
    titles = ["Original", "Noisy Input", "GAN Denoised"]
    
    for i in range(8):
        axs[0, i].imshow(sample_clean[i].cpu().squeeze(), cmap='gray')
        axs[1, i].imshow(sample_noisy[i].cpu().squeeze(), cmap='gray')
        axs[2, i].imshow(sample_denoised[i].cpu().squeeze(), cmap='gray')
        
    for row in range(3):
        axs[row, 0].set_ylabel(titles[row], fontsize=12, fontweight='bold')
        for col in range(8):
            axs[row, col].axis('off')
            
    plt.tight_layout()
    plt.savefig('denoising_demo.png', dpi=300, bbox_inches='tight')
    plt.show()