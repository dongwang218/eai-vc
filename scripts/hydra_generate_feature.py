# run_theia_inference.py
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig

import torch
from torchvision.transforms.functional import to_pil_image
import os

per_layer = True
outputs = []
def hook(module, input, output):
	if isinstance(input, tuple) and len(input) == 1 and isinstance(input[0], torch.Tensor) and not outputs:
		outputs.append(("input", "input", tuple(input[0].shape), input[0].detach().clone()))
	if isinstance(output, torch.Tensor):
		outputs.append((module.name, module.__class__.__name__, tuple(output.shape), output.detach().clone()))
def register_hook(m):
	m.register_forward_hook(hook)

@hydra.main(config_path="../vc_models/src/vc_models/conf/model", config_name="theia_vitb_sp")
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))
    orig_cwd = get_original_cwd()
    hydra_cfg = HydraConfig.get()
    # Get the config name
    config_name = hydra_cfg.job.config_name

    # Access additional params from config
    random_image_path = os.path.join(orig_cwd, cfg.get("random_image_path", "scripts/random_image.pt"))
    prev_features_path = os.path.join(orig_cwd, cfg.get("prev_features_path", "scripts/theia_features.pt"))

    model, embedding_dim, transform, metadata = instantiate(cfg)
    model.eval()
    print(model)    
    random_image = torch.load(random_image_path, map_location="cpu")
    
    if random_image.ndim == 4 and random_image.shape[-1] == 3:
        random_image = random_image.permute(0, 3, 1, 2)
        pil_images = [to_pil_image(img) for img in random_image]
    else:
        raise ValueError("random_image.pt should have shape (B,H,W,3)")

    inputs = torch.stack([transform(img) for img in pil_images])

    if per_layer:
        for name, module in model.named_modules():
            module.name = name
        model.apply(register_hook)

    with torch.no_grad():
        theia_features_new = model.forward_features(inputs)

    if per_layer:
         torch.save(
            outputs, f"{config_name}_per_layer.pt"
        )
    prev_features_data = torch.load(prev_features_path, map_location="cpu")
    if isinstance(prev_features_data, dict) and "theia_feature" in prev_features_data:
        prev_features = prev_features_data["theia_feature"]
    else:
        prev_features = prev_features_data

    if theia_features_new.shape != prev_features.shape:
        # Case 1: New is (B, N*D) and prev is (B, N, D)
        if theia_features_new.ndim == 2 and prev_features.ndim == 3:
            B, N, D = prev_features.shape
            theia_features_new = theia_features_new.reshape(B, N, D)
            print(f"Reshaped new features to: {theia_features_new.shape}")
        
        # Case 2: New is (B, N, D) and prev is (B, N*D)
        elif theia_features_new.ndim == 3 and prev_features.ndim == 2:
            B, N, D = theia_features_new.shape
            theia_features_new = theia_features_new.reshape(B, N * D)
            print(f"Flattened new features to: {theia_features_new.shape}")
        
        # Case 3: Both 3D but different shapes - flatten both
        elif theia_features_new.ndim == 3 and prev_features.ndim == 3:
            theia_features_new = theia_features_new.reshape(theia_features_new.shape[0], -1)
            prev_features = prev_features.reshape(prev_features.shape[0], -1)
            print(f"Flattened both - new: {theia_features_new.shape}, prev: {prev_features.shape}")

    # Now compare
    diff = torch.abs(theia_features_new - prev_features)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")

    if max_diff < 1e-5:
        print("✅ Features match closely!")
    else:
        print("⚠️ Features differ!")

    return theia_features_new


if __name__ == "__main__":
    main()