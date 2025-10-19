# run this in a modern conda env, such as my vjepa2-312
import transformers
from transformers import AutoModel
import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np

# deit.load_state_dict(deit_state_dict, strict=True)
# torch.save({
#     "model": deit.state_dict()},
# 	"../vc_models/src/model_ckpts/theia/theia-base-patch16-224-cdiv.pt")

random_image = torch.load("scripts/random_image.pt", map_location="cpu")
if random_image.ndim == 4 and random_image.shape[-1] == 3:
	random_image = random_image.permute(0, 3, 1, 2)
	pil_images = [to_pil_image(img) for img in random_image]
else:
	raise ValueError("random_image.pt should have shape (B,H,W,3)")


from transformers import pipeline
from transformers.image_utils import load_image

outputs = []
def hook(module, input, output):
	if isinstance(input, tuple) and len(input) == 1 and isinstance(input[0], torch.Tensor) and not outputs:
		outputs.append(("input", "input", tuple(input[0].shape), input[0].detach().clone()))
	if isinstance(output, torch.Tensor):
		outputs.append((module.name, module.__class__.__name__, tuple(output.shape), output.detach().clone()))
def register_hook(m):
	m.register_forward_hook(hook)

feature_extractor = pipeline(
    model="facebook/dinov3-vith16plus-pretrain-lvd1689m",
    task="image-feature-extraction", 
)
model = feature_extractor.model
for name, module in model.named_modules():
	module.name = name
model.apply(register_hook)

features = feature_extractor(pil_images)

config = model.config
print(model.__class__.__name__)
print(config)
arr = np.array(features[0])   # shape: (1, 201, 1280) for ViT-type backbones
tensor = torch.from_numpy(arr)  # (num_tokens, embed_dim)
tensor = tensor[:, -196:]
# 4. Save extracted feature tensors
torch.save(
	tensor,
    "scripts/dinov3_vith_features.pt"
)
torch.save(
	outputs,
    "scripts/dinov3_vith_features_per_layer.pt"
)
print("Saved features to dinov3_vith_features.pt")

# torch.save(
#     model.state_dict(),
# 	"vc_models/src/model_ckpts/dinov3/dinov3-vith16plus-pretrain-lvd1689m.pth")
