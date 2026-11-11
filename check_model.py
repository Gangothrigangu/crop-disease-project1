import torch

model_path = "models/vision_mamba_best.pth"

checkpoint = torch.load(model_path, map_location="cpu")

print("Type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nKeys:")
    for k in checkpoint.keys():
        print(k)