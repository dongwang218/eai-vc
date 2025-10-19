
import omegaconf
import hydra
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from r3m import load_r3m

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

## DEFINE PREPROCESSING
transforms = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()]) # ToTensor() divides by 255

## ENCODE IMAGE
random_image = torch.load("scripts/random_image.pt", map_location="cpu")
random_image = random_image.permute(0, 3, 1, 2)
pil_images = [to_pil_image(img) for img in random_image]
inputs = torch.stack([transforms(img) * 255.0 for img in pil_images]).to(device)
with torch.no_grad():
  embedding = r3m(inputs)
print(embedding.shape) # [1, 2048]
torch.save(
	embedding,
    "scripts/r3m_features.pt"
)

print("Saved features to r3m_features.pt")
