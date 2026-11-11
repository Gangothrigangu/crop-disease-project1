# ==============================================================================
# VISION MAMBA TRAINING SCRIPT -- PlantVillage 38-Class Dataset
# Architecture: VisionMamba(img_size=256, patch_size=16, embed_dim=192, depth=4)
# ==============================================================================
import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# -- path setup --------------------------------------------------------------
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from vision_mamba import VisionMamba

DATASET_PATH  = os.path.join(BASE_DIR, "Dataset")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
SAVED_DIR     = os.path.join(BASE_DIR, "Saved_Models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(SAVED_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_vision_mamba.pth")
CLASS_IDX_PATH  = os.path.join(MODELS_DIR, "class_indices.json")
SAVED_CI_PATH   = os.path.join(SAVED_DIR,  "mamba_class_indices.json")

# -- hyper-parameters --------------------------------------------------------
IMG_SIZE         = 256
PATCH_SIZE       = 16
EMBED_DIM        = 192
DEPTH            = 4
NUM_CLASSES      = 38
BATCH_SIZE       = 32
EPOCHS           = 15
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
LABEL_SMOOTHING  = 0.1
VAL_SPLIT        = 0.15          # 15 % held-out validation
SAMPLES_PER_CLASS = None         # None  -> use every image in the dataset
SEED             = 42
NUM_WORKERS      = 0             # safe default for Windows; raise to 2-4 if you have RAM

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -- class mapping (sorted alphabetically == ImageFolder default) -------------
classes     = sorted([d for d in os.listdir(DATASET_PATH)
                      if os.path.isdir(os.path.join(DATASET_PATH, d))])
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

print("=" * 65)
print(f"Vision Mamba Training  |  {len(classes)} classes  |  img_size={IMG_SIZE}")
print("=" * 65)
print("\nclass_to_idx (verify against class_indices.json):")
for k, v in class_to_idx.items():
    print(f"  {v:2d}: {k}")

# -- verify against existing class_indices.json -------------------------------
if os.path.exists(CLASS_IDX_PATH):
    with open(CLASS_IDX_PATH) as f:
        existing = json.load(f)
    mismatches = [(k, class_to_idx[k], existing.get(k)) for k in class_to_idx
                  if class_to_idx[k] != existing.get(k)]
    if mismatches:
        print(f"\n[WARNING] {len(mismatches)} mismatch(es) with existing class_indices.json:")
        for k, new, old in mismatches:
            print(f"  {k}: training={new}  json={old}")
    else:
        print("\n[OK] class_to_idx perfectly matches existing class_indices.json")

# save / overwrite both copies so they are always in sync with training
with open(CLASS_IDX_PATH, "w") as f:
    json.dump(class_to_idx, f, indent=4)
with open(SAVED_CI_PATH, "w") as f:
    json.dump(class_to_idx, f, indent=4)
print(f"\n[Setup] class_indices.json saved to {CLASS_IDX_PATH}")

# -- gather file paths --------------------------------------------------------
EXTS = ('.jpg', '.jpeg', '.png')

def gather_files():
    all_paths, all_labels = [], []
    for c in classes:
        c_dir   = os.path.join(DATASET_PATH, c)
        label   = class_to_idx[c]
        imgs    = sorted([f for f in os.listdir(c_dir) if f.lower().endswith(EXTS)])
        random.shuffle(imgs)
        if SAMPLES_PER_CLASS and len(imgs) > SAMPLES_PER_CLASS:
            imgs = imgs[:SAMPLES_PER_CLASS]
        for fname in imgs:
            all_paths.append(os.path.join(c_dir, fname))
            all_labels.append(label)
    return all_paths, all_labels

all_paths, all_labels = gather_files()
print(f"\n[Data] Total images gathered: {len(all_paths)}")

# -- stratified train / val split ---------------------------------------------
from collections import defaultdict
class_bins = defaultdict(list)
for p, l in zip(all_paths, all_labels):
    class_bins[l].append(p)

train_paths, train_labels = [], []
val_paths,   val_labels   = [], []

for label, paths in class_bins.items():
    n_val = max(1, int(len(paths) * VAL_SPLIT))
    val_paths   += paths[:n_val];   val_labels   += [label] * n_val
    train_paths += paths[n_val:];   train_labels += [label] * (len(paths) - n_val)

print(f"[Data] Train: {len(train_paths)}  |  Val: {len(val_paths)}")

# -- Dataset class -------------------------------------------------------------
class LeafDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths; self.labels = labels; self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.transform(img), self.labels[idx]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = LeafDataset(train_paths, train_labels, train_tf)
val_ds   = LeafDataset(val_paths,   val_labels,   val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=False)

# -- model, loss, optimiser, scheduler ----------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Setup] Device: {device}")

model = VisionMamba(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_diseases=NUM_CLASSES
)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"[Setup] VisionMamba  img_size={IMG_SIZE}  embed_dim={EMBED_DIM}  depth={DEPTH}  params={total_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Warmup (3 ep) then cosine annealing
def lr_lambda(epoch):
    warmup = 3
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, EPOCHS - warmup)
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -- training loop -------------------------------------------------------------
best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("\n" + "=" * 65)
print(f"Starting training for {EPOCHS} epochs")
print("=" * 65)
t0 = time.time()

for epoch in range(EPOCHS):
    # -- train --
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        run_loss += loss.item() * imgs.size(0)
        preds     = logits.argmax(dim=1)
        correct  += (preds == labels).sum().item()
        total    += imgs.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Ep {epoch+1}/{EPOCHS}  Batch {batch_idx+1}/{len(train_loader)}"
                  f"  loss={loss.item():.4f}", flush=True)

    train_loss = run_loss / total
    train_acc  = correct / total

    # -- validate --
    model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            v_loss    += loss.item() * imgs.size(0)
            preds      = logits.argmax(dim=1)
            v_correct += (preds == labels).sum().item()
            v_total   += imgs.size(0)

    val_loss = v_loss / v_total
    val_acc  = v_correct / v_total
    scheduler.step()
    lr_now = optimizer.param_groups[0]['lr']

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    flag = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        flag = "  <-- SAVED BEST"

    elapsed = time.time() - t0
    print(f"Epoch {epoch+1:2d}/{EPOCHS}  "
          f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
          f"lr={lr_now:.2e}{flag}  [{elapsed/60:.1f} min]", flush=True)

total_time = time.time() - t0
print(f"\n[Training] Finished in {total_time/60:.1f} min  |  Best val acc: {best_val_acc*100:.2f}%")
print(f"[Checkpoint] Saved to: {CHECKPOINT_PATH}")

# -- final evaluation on validation set (load best weights) -------------------
print("\n" + "=" * 65)
print("Final Evaluation on Validation Set (best checkpoint)")
print("=" * 65)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_true.extend(labels.numpy().tolist())

acc  = accuracy_score(all_true, all_preds)
prec = precision_score(all_true, all_preds, average='weighted', zero_division=0)
rec  = recall_score(all_true, all_preds, average='weighted', zero_division=0)
f1   = f1_score(all_true, all_preds, average='weighted', zero_division=0)

print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")

class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
print("\nPer-Class Classification Report:")
print(classification_report(all_true, all_preds, target_names=class_names, zero_division=0))

# -- save metrics JSON ---------------------------------------------------------
metrics = {
    "model": "VisionMamba",
    "img_size": IMG_SIZE, "embed_dim": EMBED_DIM, "depth": DEPTH,
    "epochs_trained": EPOCHS, "best_val_acc": round(best_val_acc, 4),
    "accuracy": round(acc, 4), "precision": round(prec, 4),
    "recall": round(rec, 4), "f1_score": round(f1, 4),
    "training_time_minutes": round(total_time / 60, 2),
}
metrics_path = os.path.join(RESULTS_DIR, "mamba_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"\n[Results] Metrics saved to {metrics_path}")

# -- per-class sample predictions (10 random images per class) ----------------
print("\n" + "=" * 65)
print("Sample Predictions -- 3 images from each class")
print("=" * 65)

sample_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

wrong_count = 0
total_samples = 0
model.eval()
for cls_name, cls_idx in class_to_idx.items():
    cls_dir   = os.path.join(DATASET_PATH, cls_name)
    imgs_list = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(EXTS)])
    sample_imgs = random.sample(imgs_list, min(3, len(imgs_list)))
    local_correct = 0
    for fname in sample_imgs:
        fpath = os.path.join(cls_dir, fname)
        try:
            img = Image.open(fpath).convert('RGB')
            tensor = sample_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
                pred_idx = probs.argmax().item()
                conf     = probs[pred_idx].item() * 100
            pred_name = idx_to_class[pred_idx]
            match = "OK" if pred_idx == cls_idx else "X"
            if pred_idx != cls_idx:
                wrong_count += 1
            else:
                local_correct += 1
            total_samples += 1
        except Exception as e:
            pred_name = f"ERROR({e})"; match = "X"; conf = 0.0
    print(f"  [{cls_idx:2d}] {cls_name:<55s}  {local_correct}/{len(sample_imgs)} correct")

print(f"\nSample test: {total_samples - wrong_count}/{total_samples} correct "
      f"({(total_samples-wrong_count)/total_samples*100:.1f}%)")
print("\n[Done] Vision Mamba training and evaluation complete.")

# -- copy checkpoint to Saved_Models as well ----------------------------------
import shutil
SAVED_CHECKPOINT = os.path.join(SAVED_DIR, "best_vision_mamba.pth")
shutil.copy2(CHECKPOINT_PATH, SAVED_CHECKPOINT)
print(f"\n[Checkpoint] Copied to Saved_Models: {SAVED_CHECKPOINT}")

# -- web-app load verification -------------------------------------------------
print("\n" + "=" * 65)
print("Web App Load Verification")
print("=" * 65)
try:
    verify_model = VisionMamba(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM, depth=DEPTH,
        num_diseases=NUM_CLASSES
    )
    sd = torch.load(CHECKPOINT_PATH, map_location='cpu')
    missing, unexpected = verify_model.load_state_dict(sd, strict=True)
    verify_model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = verify_model(dummy)
    assert out.shape == (1, NUM_CLASSES), f"Unexpected output shape: {out.shape}"
    print(f"[OK] Checkpoint loads cleanly. Output shape: {out.shape}")
    print(f"[OK] Missing keys  : {missing}")
    print(f"[OK] Unexpected keys: {unexpected}")
except Exception as ve:
    print(f"[ERROR] Web app load verification failed: {ve}")

print("\n[All Done] Vision Mamba retrain complete. CNN and UI were NOT modified.")

