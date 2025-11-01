# src/test_dataset.py
from dataset import get_dataloader
import matplotlib.pyplot as plt

# Load dataloader
loader = get_dataloader(batch_size=1, num_workers=0)
covers, secrets = next(iter(loader))

# Convert tensors to images for display
cover_img = covers[0].permute(1,2,0).numpy()
secret_img = secrets[0].permute(1,2,0).numpy()

# Display images
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(cover_img)
plt.title("Cover Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(secret_img)
plt.title("Secret Image")
plt.axis('off')

plt.show()
