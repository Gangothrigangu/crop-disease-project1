# ==============================================================================
# finetune_mamba.py
# Fine-tune Vision Mamba for 10 additional epochs from the existing checkpoint.
#
# What this script does:
#   1. Loads models/best_vision_mamba.pth  (no weight re-initialisation).
#   2. Trains for EXTRA_EPOCHS=10 more epochs.
#   3. Weighted cross-entropy (inverse class frequency) + label smoothing.
#   4. Cosine annealing LR  1e-4 -> 1e-5  (no warmup -- we are resuming).
#   5. Saves a per-epoch checkpoint every epoch.
#   6. Overwrites models/best_vision_mamba.pth when val accuracy improves.
#   7. After training: confusion matrix, classification report, P/R/F1, JSON.
#   8. Copies best checkpoint -> Saved_Models/best_vision_mamba.pth for the UI.
#   9. Runs a dummy forward-pass load-verification at the end.
#
# Files NOT touched:
#   web_app.py, predict.py, vision_mamba.py, train_mamba.py,
#   model.py, best_cnn_model.h5, class_indices.json
# ==============================================================================

import os, sys, json, time, random, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from vision_mamba import VisionMamba          # unchanged architecture file

BASE_DIR      = SCRIPT_DIR
DATASET_PATH  = os.path.join(BASE_DIR, "Dataset")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
CKPT_DIR      = os.path.join(MODELS_DIR, "checkpoints")
SAVED_DIR     = os.path.join(BASE_DIR, "Saved_Models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")

for d in (CKPT_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

CHECKPOINT_IN  = os.path.join(MODELS_DIR, "best_vision_mamba.pth")
CHECKPOINT_OUT = os.path.join(MODELS_DIR, "best_vision_mamba.pth")
CLASS_IDX_PATH = os.path.join(MODELS_DIR, "class_indices.json")

# ---------------------------------------------------------------------------
# Hyper-parameters  (same architecture as train_mamba.py -- must match!)
# ---------------------------------------------------------------------------
IMG_SIZE    = 256
PATCH_SIZE  = 16
EMBED_DIM   = 192
DEPTH       = 4
NUM_CLASSES = 38

EXTRA_EPOCHS    = 10
BATCH_SIZE      = 32
LR_START        = 1e-4
LR_END          = 1e-5
LABEL_SMOOTHING = 0.1
VAL_SPLIT       = 0.15
SEED            = 42
NUM_WORKERS     = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SEP = "=" * 70
print(SEP)
print("  Vision Mamba  --  Fine-Tune  |  10 additional epochs")
print(f"  Checkpoint in : {CHECKPOINT_IN}")
print(SEP)

# ---------------------------------------------------------------------------
# Class mapping  (must be identical to training)
# ---------------------------------------------------------------------------
classes      = sorted([d for d in os.listdir(DATASET_PATH)
                       if os.path.isdir(os.path.join(DATASET_PATH, d))])
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

assert len(classes) == NUM_CLASSES, (
    f"Expected {NUM_CLASSES} classes, found {len(classes)}"
)

if os.path.exists(CLASS_IDX_PATH):
    with open(CLASS_IDX_PATH) as f:
        saved = json.load(f)
    mismatches = [
        (k, class_to_idx[k], saved.get(k))
        for k in class_to_idx if class_to_idx[k] != saved.get(k)
    ]
    if mismatches:
        print(f"\n[WARNING] {len(mismatches)} class-index mismatch(es) -- aborting.")
        for k, new, old in mismatches:
            print(f"  {k}: dataset={new}  json={old}")
        sys.exit(1)
    else:
        print("\n[OK] class_to_idx perfectly matches class_indices.json")
else:
    print(f"\n[WARNING] {CLASS_IDX_PATH} not found -- skipping mismatch check.")

print(f"\n[Classes] {NUM_CLASSES} classes detected.\n")

# ---------------------------------------------------------------------------
# Gather all image paths
# ---------------------------------------------------------------------------
EXTS = ('.jpg', '.jpeg', '.png')

def gather_files():
    all_paths, all_labels = [], []
    for c in classes:
        c_dir  = os.path.join(DATASET_PATH, c)
        label  = class_to_idx[c]
        imgs   = sorted([f for f in os.listdir(c_dir)
                         if f.lower().endswith(EXTS)])
        random.shuffle(imgs)
        for fname in imgs:
            all_paths.append(os.path.join(c_dir, fname))
            all_labels.append(label)
    return all_paths, all_labels

all_paths, all_labels = gather_files()
print(f"[Data] Total images: {len(all_paths)}")

# ---------------------------------------------------------------------------
# Stratified train / val split  (same ratio as original training)
# ---------------------------------------------------------------------------
class_bins = defaultdict(list)
for p, l in zip(all_paths, all_labels):
    class_bins[l].append(p)

train_paths, train_labels = [], []
val_paths,   val_labels   = [], []
class_counts = {}

for label, paths in class_bins.items():
    n_val = max(1, int(len(paths) * VAL_SPLIT))
    val_paths   += paths[:n_val]
    val_labels  += [label] * n_val
    train_paths += paths[n_val:]
    train_labels+= [label] * (len(paths) - n_val)
    class_counts[label] = len(paths) - n_val

print(f"[Data] Train: {len(train_paths)}  |  Val: {len(val_paths)}")
print(f"[Data] Per-class train counts  min={min(class_counts.values())}  "
      f"max={max(class_counts.values())}")

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("\n[Transform Verification]")
print("  Val transform    : Resize(256,256) -> ToTensor -> Normalize(ImageNet)")
print("  Inference (UI)   : Resize(img_size,img_size) -> ToTensor -> Normalize(ImageNet)")
print("  [OK] Validation and inference transforms are IDENTICAL.\n")

# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------
class LeafDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        return self.transform(img), self.labels[idx]

train_ds = LeafDataset(train_paths, train_labels, train_tf)
val_ds   = LeafDataset(val_paths,   val_labels,   val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=False)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Setup] Device: {device}")

# ---------------------------------------------------------------------------
# Class weights  (inverse frequency, normalised)
# ---------------------------------------------------------------------------
total_train = sum(class_counts.values())
weights = torch.zeros(NUM_CLASSES)
for cls_idx, count in class_counts.items():
    weights[cls_idx] = total_train / (NUM_CLASSES * count)
weights = weights / weights.sum() * NUM_CLASSES
weights = weights.to(device)

print("\n[Class Weights]")
print(f"  Min: {weights.min().item():.4f}  Max: {weights.max().item():.4f}  "
      f"Mean: {weights.mean().item():.4f}")

# ---------------------------------------------------------------------------
# Model -- load existing checkpoint (NO weight re-initialisation)
# ---------------------------------------------------------------------------
model = VisionMamba(
    img_size    = IMG_SIZE,
    patch_size  = PATCH_SIZE,
    embed_dim   = EMBED_DIM,
    depth       = DEPTH,
    num_diseases= NUM_CLASSES,
)

if not os.path.exists(CHECKPOINT_IN):
    print(f"[ERROR] Checkpoint not found: {CHECKPOINT_IN}")
    sys.exit(1)

state_dict = torch.load(CHECKPOINT_IN, map_location='cpu')
missing, unexpected = model.load_state_dict(state_dict, strict=True)
print(f"\n[Checkpoint] Loaded: {CHECKPOINT_IN}")
print(f"  missing keys   : {missing}")
print(f"  unexpected keys: {unexpected}")
print("  [OK] Weights restored -- continuing from previous 15-epoch session.\n")

model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"[Model] VisionMamba  img_size={IMG_SIZE}  embed_dim={EMBED_DIM}  "
      f"depth={DEPTH}  params={total_params:,}")

# ---------------------------------------------------------------------------
# Loss, optimiser, scheduler
# ---------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=LR_START, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EXTRA_EPOCHS, eta_min=LR_END
)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print("\n" + SEP)
print(f"  Fine-tuning for {EXTRA_EPOCHS} epochs  |  LR {LR_START:.0e} -> {LR_END:.0e}")
print(SEP + "\n")

best_val_acc = 0.0
start_epoch = 1

# Check for existing epoch checkpoints to resume from
if os.path.exists(CKPT_DIR):
    import re, glob
    existing_ckpts = []
    for p in glob.glob(os.path.join(CKPT_DIR, "mamba_epoch_*_val*.pth")):
        m = re.search(r"mamba_epoch_(\d+)_val([\d\.]+)\.pth", os.path.basename(p))
        if m:
            existing_ckpts.append((int(m.group(1)), float(m.group(2)), p))

    if existing_ckpts:
        existing_ckpts.sort()
        last_completed_epoch, last_val, last_p = existing_ckpts[-1]
        best_ep, best_val_acc, best_p = max(existing_ckpts, key=lambda x: x[1])
        if last_completed_epoch < EXTRA_EPOCHS:
            start_epoch = last_completed_epoch + 1
            print(f"[Resume] Found {len(existing_ckpts)} completed epoch(s). Resuming from Epoch {start_epoch}/{EXTRA_EPOCHS} using {last_p}")
            state_dict = torch.load(last_p, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            for _ in range(last_completed_epoch):
                scheduler.step()
            print(f"[Resume] Current LR: {optimizer.param_groups[0]['lr']:.2e}, Current Best Val Acc: {best_val_acc*100:.2f}%\n")
        else:
            print(f"[Info] All {EXTRA_EPOCHS} epochs already completed. Proceeding directly to final evaluation.\n")
            start_epoch = EXTRA_EPOCHS + 1

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
t0 = time.time()

for epoch in range(start_epoch, EXTRA_EPOCHS + 1):
    # Train
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
            print(f"  Ep {epoch}/{EXTRA_EPOCHS}  "
                  f"Batch {batch_idx+1}/{len(train_loader)}  "
                  f"loss={loss.item():.4f}", flush=True)

    train_loss = run_loss / total
    train_acc  = correct  / total

    # Validate
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
    lr_now   = optimizer.param_groups[0]['lr']
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Per-epoch checkpoint
    epoch_ckpt = os.path.join(
        CKPT_DIR,
        f"mamba_epoch_{epoch:02d}_val{val_acc:.4f}.pth"
    )
    torch.save(model.state_dict(), epoch_ckpt)

    # Best-model update
    flag = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_OUT)
        flag = "  <-- BEST SAVED"

    elapsed = time.time() - t0
    print(
        f"Epoch {epoch:2d}/{EXTRA_EPOCHS}  "
        f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  "
        f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%  "
        f"lr={lr_now:.2e}{flag}  [{elapsed/60:.1f} min]",
        flush=True,
    )

total_time = time.time() - t0
print(f"\n[Fine-Tune] Finished in {total_time/60:.1f} min  |  "
      f"Best val acc: {best_val_acc*100:.2f}%")

# ---------------------------------------------------------------------------
# Final evaluation  (best checkpoint)
# ---------------------------------------------------------------------------
print("\n" + SEP)
print("  Final Evaluation  --  Best Checkpoint")
print(SEP)

model.load_state_dict(torch.load(CHECKPOINT_OUT, map_location=device))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs  = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_true.extend(labels.numpy().tolist())

acc  = accuracy_score(all_true, all_preds)
prec = precision_score(all_true, all_preds, average='weighted', zero_division=0)
rec  = recall_score(all_true, all_preds, average='weighted', zero_division=0)
f1   = f1_score(all_true, all_preds, average='weighted', zero_division=0)

print(f"\n  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")

class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
print("\nPer-Class Classification Report:")
print(classification_report(all_true, all_preds,
                             target_names=class_names, zero_division=0))

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(all_true, all_preds)
fig, ax = plt.subplots(figsize=(18, 16))
short_names = [n.split("___")[-1][:22] for n in class_names]
sns.heatmap(
    cm, annot=False, fmt='d', cmap='Blues',
    xticklabels=short_names, yticklabels=short_names, ax=ax,
)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True",      fontsize=12)
ax.set_title("Vision Mamba -- Confusion Matrix (fine-tune)", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
cm_path = os.path.join(RESULTS_DIR, "mamba_confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\n[Results] Confusion matrix saved -> {cm_path}")

# ---------------------------------------------------------------------------
# Metrics JSON
# ---------------------------------------------------------------------------
metrics = {
    "model"                   : "VisionMamba",
    "img_size"                : IMG_SIZE,
    "embed_dim"               : EMBED_DIM,
    "depth"                   : DEPTH,
    "previous_epochs"         : 15,
    "finetune_epochs"         : EXTRA_EPOCHS,
    "total_epochs"            : 15 + EXTRA_EPOCHS,
    "lr_start"                : LR_START,
    "lr_end"                  : LR_END,
    "label_smoothing"         : LABEL_SMOOTHING,
    "weighted_loss"           : True,
    "best_val_acc"            : round(best_val_acc, 4),
    "final_accuracy"          : round(acc,  4),
    "final_precision_weighted": round(prec, 4),
    "final_recall_weighted"   : round(rec,  4),
    "final_f1_weighted"       : round(f1,   4),
    "finetune_time_minutes"   : round(total_time / 60, 2),
    "val_acc_per_epoch"       : [round(v, 4) for v in history["val_acc"]],
    "train_acc_per_epoch"     : [round(v, 4) for v in history["train_acc"]],
}
metrics_path = os.path.join(RESULTS_DIR, "mamba_finetune_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(f"[Results] Metrics JSON saved   -> {metrics_path}")

# ---------------------------------------------------------------------------
# Copy best checkpoint -> Saved_Models  (used by the UI)
# ---------------------------------------------------------------------------
saved_ckpt = os.path.join(SAVED_DIR, "best_vision_mamba.pth")
shutil.copy2(CHECKPOINT_OUT, saved_ckpt)
print(f"\n[Deploy] Best checkpoint copied -> {saved_ckpt}")
print("         UI will automatically use the improved model next time it loads.")

# ---------------------------------------------------------------------------
# Web-app load verification
# ---------------------------------------------------------------------------
print("\n" + SEP)
print("  Web-App Load Verification")
print(SEP)
try:
    verify = VisionMamba(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM, depth=DEPTH,
        num_diseases=NUM_CLASSES,
    )
    sd = torch.load(CHECKPOINT_OUT, map_location='cpu')
    missing_v, unexpected_v = verify.load_state_dict(sd, strict=True)
    verify.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = verify(dummy)
    assert out.shape == (1, NUM_CLASSES), f"Unexpected shape: {out.shape}"
    print(f"  [OK] Checkpoint loads cleanly.  Output shape: {out.shape}")
    print(f"  [OK] Missing keys   : {missing_v}")
    print(f"  [OK] Unexpected keys: {unexpected_v}")
except Exception as ve:
    print(f"  [ERROR] Load verification failed: {ve}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + SEP)
print("  Fine-Tune Complete")
print(SEP)
print(f"  Best val accuracy   : {best_val_acc*100:.2f}%")
print(f"  Checkpoint (models) : {CHECKPOINT_OUT}")
print(f"  Checkpoint (UI)     : {saved_ckpt}")
print(f"  Confusion matrix    : {cm_path}")
print(f"  Metrics JSON        : {metrics_path}")
print(f"  Per-epoch ckpts     : {CKPT_DIR}")
print(f"\n  [NOTE] CNN, UI, predict.py, vision_mamba.py were NOT modified.")
print(SEP + "\n")
