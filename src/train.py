import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CustomCelebAPairs
from torch.utils.data import DataLoader, Subset
from models import Encoder, Decoder
from torchvision.utils import save_image
import os
from math import log10
from pytorch_msssim import ssim  # pip install pytorch-msssim

# =====================
# Utility functions
# =====================
def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * log10(1.0 / torch.sqrt(mse))  # assuming input in [0,1]

# =====================
# Main
# =====================
if __name__ == "__main__":
    # =====================
    # Hyperparameters
    # =====================
    batch_size = 1
    lr = 1e-3
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # =====================
    # Load Data
    # =====================
    full_dataset = CustomCelebAPairs()
    start_idx = 146000
    end_idx = 147000
    currepoch = 13
    subset_dataset = Subset(full_dataset, list(range(start_idx, end_idx)))
    loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # =====================
    # Initialize Models
    # =====================
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    # =====================
    # Load Previous Checkpoint
    # =====================
    checkpoint_path = os.path.join(save_dir, f"model_epoch_{currepoch}.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"Loaded previous model_epoch_{currepoch}.pth")

    # =====================
    # Loss and Optimizer
    # =====================
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    if os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # =====================
    # Training Loop
    # =====================
    for epoch in range(num_epochs):
        total_loss, total_psnr_stego, total_psnr_secret = 0, 0, 0
        total_ssim_stego, total_ssim_secret = 0, 0

        for i, (cover, secret) in enumerate(loader):
            cover = cover.to(device)
            secret = secret.to(device)

            # Forward pass
            stego = encoder(cover, secret)
            recovered = decoder(stego)

            # Loss
            # Hybrid Loss (MSE + SSIM)
            loss_cover_mse = criterion(stego, cover)
            loss_secret_mse = criterion(recovered, secret)

            # SSIM returns similarity → (1 - SSIM) is dissimilarity (acts as loss)
            loss_cover_ssim = 1 - ssim(stego, cover, data_range=1.0)
            loss_secret_ssim = 1 - ssim(recovered, secret, data_range=1.0)

            # Weighted combination (tune 0.5 based on how much you want SSIM to matter)
            loss = (loss_cover_mse + loss_secret_mse) #+ 0.5 * (loss_cover_ssim + loss_secret_ssim)


            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute metrics
            psnr_stego = compute_psnr(stego.detach(), cover)
            psnr_secret = compute_psnr(recovered.detach(), secret)
            ssim_stego = ssim(stego, cover, data_range=1.0).item()
            ssim_secret = ssim(recovered, secret, data_range=1.0).item()

            total_loss += loss.item()
            total_psnr_stego += psnr_stego
            total_psnr_secret += psnr_secret
            total_ssim_stego += ssim_stego
            total_ssim_secret += ssim_secret

            # Print batch progress
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(loader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"PSNR(Stego): {psnr_stego:.2f}, SSIM(Stego): {ssim_stego:.4f}, "
                    f"PSNR(Secret): {psnr_secret:.2f}, SSIM(Secret): {ssim_secret:.4f}"
                )

        # Epoch summary
        n_batches = len(loader)
        print("\nEpoch Summary:")
        print(f"Avg Loss: {total_loss/n_batches:.4f}")
        print(f"Avg PSNR (Stego): {total_psnr_stego/n_batches:.2f}, Avg SSIM (Stego): {total_ssim_stego/n_batches:.4f}")
        print(f"Avg PSNR (Secret): {total_psnr_secret/n_batches:.2f}, Avg SSIM (Secret): {total_ssim_secret/n_batches:.4f}\n")

        # Save model checkpoint
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, f"model_epoch_{currepoch+1}.pth"))  # next epoch number

    print("Training Completed!")
