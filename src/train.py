# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CustomCelebAPairs
from torch.utils.data import DataLoader, Subset
from models import Encoder, Decoder
import os

if __name__ == "__main__":
    # =====================
    # Hyperparameters
    # =====================
    batch_size = 64
    lr = 1e-3
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # =====================
    # Load Data (next 11k images)
    # =====================
    full_dataset = CustomCelebAPairs()
    start_idx = 95000
    end_idx = 105000
    currepoch = 9
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
        total_loss = 0
        for i, (cover, secret) in enumerate(loader):
            cover = cover.to(device)
            secret = secret.to(device)

            # Forward pass
            stego = encoder(cover, secret)
            recovered = decoder(stego)

            # Loss
            loss_cover = criterion(stego, cover)
            loss_secret = criterion(recovered, secret)
            loss = loss_cover + loss_secret

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i+1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {total_loss/len(loader):.4f}")

        # Save model checkpoint
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, f"model_epoch_{currepoch+1}.pth"))  # next epoch number

    print("Training Completed!")