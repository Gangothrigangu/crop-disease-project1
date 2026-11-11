import os
import json

dataset_path = r"Dataset"

classes = sorted([
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
])

class_indices = {cls: idx for idx, cls in enumerate(classes)}

print("Total Classes:", len(class_indices))

os.makedirs("models", exist_ok=True)

with open("models/class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=4)

print("class_indices.json created successfully")