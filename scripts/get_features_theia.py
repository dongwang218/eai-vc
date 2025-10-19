# run this in a modern conda env, such as my vjepa2-312
import transformers
from transformers import AutoModel
import torch
model = AutoModel.from_pretrained("theaiinstitute/theia-base-patch16-224-cdiv", trust_remote_code=True)

deit = AutoModel.from_pretrained("facebook/deit-base-patch16-224", image_size=224)
deit.pooler = torch.nn.Identity()
deit_state_dict = {k.replace("model.", ""): v for k, v in model.backbone.state_dict().items()}

# 4. Load into Theia’s model (ignore missing heads or extra keys)
deit.load_state_dict(deit_state_dict, strict=True)
torch.save({
    "model": deit.state_dict()},
	"../vc_models/src/model_ckpts/theia/theia-base-patch16-224-cdiv.pt")

random_image = torch.load("random_image.pt", map_location="cpu")
with torch.no_grad():
	# Theia / intermediate feature, mainly used for robot learning.
	# To change different feature reduction methods, pass `feature_reduction_method` argument in AutoModel.from_pretrained() method
    theia_feature = model.forward_feature(random_image)

# 4. Save extracted feature tensors
torch.save(
	theia_feature,
    "theia_features.pt"
)

print("Saved features to theia_features.pt")
