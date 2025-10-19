import torch

# Create a random RGB image with values in [0, 255], dtype=uint8
random_image = torch.randint(
    low=0, 
    high=256, 
    size=(1, 224, 224, 3),  # (batch, height, width, channels)
    dtype=torch.uint8
)

# Save using a format that works across torch versions
torch.save(random_image, "random_image.pt")

print("Saved random image tensor to random_image.pt")

