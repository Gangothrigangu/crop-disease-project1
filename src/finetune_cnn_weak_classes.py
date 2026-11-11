# ==============================================================================
# CNN TARGETED FINE-TUNING v2 — WEAK CLASS FIX (CORRECTED)
# Changes from v1:
#   - Removed slow per-sample Python augmentation loop (was causing oscillation)
#   - Uses tf ImageDataGenerator.flow() directly — faster and more stable
#   - LR raised to 1e-4 (was 5e-5) to escape the plateau
#   - Class weight boost reduced to 1.4x (was 1.8x) to prevent oscillation
#   - Weak classes get 300 samples, strong classes get 150 samples
# Does NOT touch Vision Mamba files.
# ==============================================================================

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATASET_PATH     = os.path.join(BASE_DIR, "Dataset")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "Saved_Models")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR,       exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,      exist_ok=True)

BEST_MODEL_PATH     = os.path.join(MODELS_DIR,       "best_cnn_model.h5")
FALLBACK_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "cnn_model.h5")
CLASS_INDEX_PATH    = os.path.join(MODELS_DIR,        "class_indices.json")
if not os.path.exists(CLASS_INDEX_PATH):
    CLASS_INDEX_PATH = os.path.join(SAVED_MODELS_DIR, "class_indices.json")

# ---------------------------------------------------------------------------
# Weak classes (from confusion-matrix analysis)
# ---------------------------------------------------------------------------
WEAK_CLASSES = {
    "Tomato___healthy",
    "Tomato___Target_Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Pepper,_bell___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Early_blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Grape___Black_rot",
    "Grape___healthy",
    "Peach___healthy",
    "Blueberry___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Tomato___Tomato_mosaic_virus",
}

# Hyper-parameters
FINETUNE_EPOCHS          = 5
FINETUNE_LR              = 1e-4     # raised from 5e-5 to escape plateau
BATCH_SIZE               = 32
IMG_SIZE                 = 128
SEED                     = 42
WEAK_SAMPLES_PER_CLASS   = 300
STRONG_SAMPLES_PER_CLASS = 150
CLASS_WEIGHT_BOOST       = 1.4      # reduced from 1.8 to prevent oscillation


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image(fpath):
    img = tf.keras.preprocessing.image.load_img(fpath, target_size=(IMG_SIZE, IMG_SIZE))
    return tf.keras.preprocessing.image.img_to_array(img) / 255.0


def load_dataset(dataset_path, class_to_idx, val_split=0.2):
    np.random.seed(SEED)
    train_x, train_y = [], []
    val_x,   val_y   = [], []

    classes = sorted(class_to_idx.keys())
    print(f"\n[Data] Loading dataset — {len(classes)} classes")

    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] Missing: {cls_dir}")
            continue

        imgs = sorted([f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        np.random.shuffle(imgs)

        n = WEAK_SAMPLES_PER_CLASS if cls in WEAK_CLASSES else STRONG_SAMPLES_PER_CLASS
        selected = imgs[:n]
        n_val    = max(1, int(len(selected) * val_split))
        val_fs   = selected[:n_val]
        train_fs = selected[n_val:]

        label = class_to_idx[cls]
        tag   = "WEAK  " if cls in WEAK_CLASSES else "strong"

        t_loaded = 0
        for fn in train_fs:
            try:
                train_x.append(load_image(os.path.join(cls_dir, fn)))
                train_y.append(label)
                t_loaded += 1
            except Exception:
                continue

        v_loaded = 0
        for fn in val_fs:
            try:
                val_x.append(load_image(os.path.join(cls_dir, fn)))
                val_y.append(label)
                v_loaded += 1
            except Exception:
                continue

        print(f"  [{tag}] {cls:<55s}  train={t_loaded:>3}  val={v_loaded:>2}")

    X_tr = np.array(train_x, dtype=np.float32)
    y_tr = np.array(train_y, dtype=np.int32)
    X_va = np.array(val_x,   dtype=np.float32)
    y_va = np.array(val_y,   dtype=np.int32)
    print(f"\n[Data] Train: {len(X_tr)}  Val: {len(X_va)}")
    return X_tr, y_tr, X_va, y_va


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_cnn_model(num_classes):
    from model import create_model
    for path in (BEST_MODEL_PATH, FALLBACK_MODEL_PATH):
        if not os.path.exists(path):
            continue
        print(f"[Model] Found checkpoint: {path}")
        try:
            m = create_model(num_classes)
            m.load_weights(path)
            print("[Model] Loaded via load_weights()")
            return m, path
        except Exception as e1:
            print(f"[Model] load_weights failed ({e1}), trying load_model...")
            try:
                m = tf.keras.models.load_model(path, compile=False)
                print("[Model] Loaded via load_model()")
                return m, path
            except Exception as e2:
                print(f"[Model] load_model also failed ({e2}), skipping {path}")
    raise FileNotFoundError(
        "No CNN checkpoint found. Run train.py first.\n"
        f"  Tried: {BEST_MODEL_PATH}\n  Tried: {FALLBACK_MODEL_PATH}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def finetune(epochs=FINETUNE_EPOCHS, batch_size=BATCH_SIZE):
    start = time.time()

    print("=" * 65)
    print("CNN TARGETED FINE-TUNING v2 — WEAK CLASS FIX")
    print(f"Epochs: {epochs}  |  LR: {FINETUNE_LR}  |  Batch: {batch_size}")
    print(f"WeightBoost: {CLASS_WEIGHT_BOOST}x for {len(WEAK_CLASSES)} weak classes")
    print("=" * 65)

    # 1. Class mapping
    with open(CLASS_INDEX_PATH) as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes  = len(class_names)
    weak_indices = {class_to_idx[c] for c in WEAK_CLASSES if c in class_to_idx}

    print(f"\n[Setup] Classes: {num_classes}  |  Weak: {len(weak_indices)}")

    # 2. Data
    X_tr, y_tr, X_va, y_va = load_dataset(DATASET_PATH, class_to_idx)

    # 3. Class weights
    classes_present = np.unique(y_tr)
    w_arr = compute_class_weight('balanced', classes=classes_present, y=y_tr)
    cw = {int(c): float(w) for c, w in zip(classes_present, w_arr)}
    for idx in weak_indices:
        if idx in cw:
            cw[idx] *= CLASS_WEIGHT_BOOST

    print(f"\n[Weights] Weak-class weights (boosted ×{CLASS_WEIGHT_BOOST}):")
    for idx in sorted(weak_indices):
        if idx in cw:
            print(f"  [{idx:2d}] {idx_to_class.get(idx, ''):<55s}  {cw[idx]:.3f}")

    # 4. Load model
    model, loaded_from = load_cnn_model(num_classes)

    # 5. Recompile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"\n[Model] Recompiled  Adam(lr={FINETUNE_LR})")

    # 6. Baseline
    print("\n[Baseline] Evaluating current CNN before fine-tuning...")
    y_base_pred = np.argmax(model.predict(X_va, batch_size=batch_size, verbose=0), axis=1)
    base_acc    = accuracy_score(y_va, y_base_pred)
    base_f1_pc  = f1_score(y_va, y_base_pred, average=None, zero_division=0)
    print(f"[Baseline] Accuracy: {base_acc*100:.2f}%")

    # 7. Augmentation generator (single generator, applied uniformly)
    aug_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=25,
        zoom_range=0.20,
        horizontal_flip=True,
        vertical_flip=False,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.10,
        brightness_range=[0.80, 1.25],
        fill_mode='nearest',
    )
    train_flow = aug_gen.flow(X_tr, y_tr, batch_size=batch_size, shuffle=True, seed=SEED)

    # 8. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    # 9. Train
    print(f"\n[Training] Starting fine-tuning for {epochs} epochs...")
    history = model.fit(
        train_flow,
        validation_data=(X_va, y_va),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=cw,
    )

    elapsed = time.time() - start
    print(f"\n[Training] Finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # 10. Sync best model
    if os.path.exists(BEST_MODEL_PATH):
        try:
            best = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
            best.save(FALLBACK_MODEL_PATH)
            print(f"[Model] Best checkpoint synced → {FALLBACK_MODEL_PATH}")
            model = best
        except Exception as e:
            print(f"[WARN] Sync failed: {e}")

    # 11. Final evaluation
    print("\n[Eval] Evaluating fine-tuned CNN...")
    y_probs  = model.predict(X_va, batch_size=batch_size, verbose=0)
    y_pred   = np.argmax(y_probs, axis=1)

    acc  = float(accuracy_score(y_va, y_pred))
    prec = float(precision_score(y_va, y_pred, average='weighted', zero_division=0))
    rec  = float(recall_score(y_va,   y_pred, average='weighted', zero_division=0))
    f1   = float(f1_score(y_va,       y_pred, average='weighted', zero_division=0))
    new_f1_pc = f1_score(y_va, y_pred, average=None, zero_division=0)

    # 12. Confusion matrix
    cm = confusion_matrix(y_va, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'CNN After Fine-Tuning (Acc: {acc*100:.2f}%)', fontsize=14)
    plt.ylabel('Actual', fontsize=11)
    plt.xlabel('Predicted', fontsize=11)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Results] Confusion matrix → {cm_path}")

    # 13. Classification report
    cls_rep  = classification_report(y_va, y_pred, target_names=class_names, zero_division=0)
    rep_path = os.path.join(RESULTS_DIR, "classification_report_cnn.txt")
    with open(rep_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write(f"CNN CLASSIFICATION REPORT — Fine-Tuned v2 ({epochs} epochs)\n")
        f.write("=" * 65 + "\n\n")
        f.write(cls_rep)
        f.write(f"\nFine-Tune Time : {elapsed:.2f}s\n")
        f.write(f"Loaded from    : {loaded_from}\n")
    print(f"[Results] Classification report → {rep_path}")

    # 14. Before/After comparison table
    print("\n" + "=" * 85)
    print("WEAK CLASS — BEFORE vs AFTER F1 COMPARISON")
    print("=" * 85)
    print(f"{'Class':<55} {'Before':>8} {'After':>8} {'Delta':>8}")
    print("-" * 85)
    for idx in sorted(weak_indices):
        name   = idx_to_class.get(idx, str(idx))
        before = base_f1_pc[idx] if idx < len(base_f1_pc) else 0.0
        after  = new_f1_pc[idx]  if idx < len(new_f1_pc)  else 0.0
        delta  = after - before
        arrow  = "▲" if delta > 0.02 else ("▼" if delta < -0.02 else "─")
        print(f"{name:<55} {before:>8.3f} {after:>8.3f} {arrow}{delta:>+7.3f}")

    print("\n--- Strong class spot-check (first 5) ---")
    strong_idx = [i for i in range(num_classes) if i not in weak_indices]
    for idx in strong_idx[:5]:
        name   = idx_to_class.get(idx, str(idx))
        before = base_f1_pc[idx] if idx < len(base_f1_pc) else 0.0
        after  = new_f1_pc[idx]  if idx < len(new_f1_pc)  else 0.0
        delta  = after - before
        arrow  = "▲" if delta > 0.02 else ("▼" if delta < -0.02 else "─")
        print(f"{name:<55} {before:>8.3f} {after:>8.3f} {arrow}{delta:>+7.3f}")

    # 15. Metrics JSON
    metrics = {
        "model": "CNN",
        "mode": "fine-tuned-v2 (weak class fix)",
        "epochs_finetuned": epochs,
        "fine_tune_lr": FINETUNE_LR,
        "fine_tune_time_seconds": round(elapsed, 2),
        "baseline_accuracy_before": round(base_acc, 4),
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "support":   int(len(y_va))
    }
    json_path = os.path.join(RESULTS_DIR, "cnn_metrics.json")
    if os.path.exists(json_path):
        try:
            with open(json_path) as jf:
                old = json.load(jf)
                metrics["original_training_time_seconds"] = old.get("training_time_seconds", 0.0)
                metrics["original_epochs"] = old.get("epochs", 10)
        except Exception:
            pass
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n[Results] Metrics JSON → {json_path}")

    # 16. Accuracy/loss curves
    ep = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(ep, history.history['accuracy'],     'b-o', label='Train Acc')
    plt.plot(ep, history.history['val_accuracy'], 'g-s', label='Val Acc')
    plt.axhline(base_acc, color='gray', ls='--', label=f'Baseline {base_acc*100:.1f}%')
    plt.title('CNN Fine-Tune v2 — Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_finetune_accuracy_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(ep, history.history['loss'],     'b-o', label='Train Loss')
    plt.plot(ep, history.history['val_loss'], 'r-s', label='Val Loss')
    plt.title('CNN Fine-Tune v2 — Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cnn_finetune_loss_curve.png"), dpi=300)
    plt.close()

    # 17. Summary
    print("\n" + "=" * 55)
    print("CNN FINE-TUNE v2 — FINAL SUMMARY")
    print("=" * 55)
    print(f"Accuracy  before : {base_acc * 100:.2f}%")
    print(f"Accuracy  after  : {acc  * 100:.2f}%  (Δ {(acc-base_acc)*100:+.2f}%)")
    print(f"Precision after  : {prec * 100:.2f}%")
    print(f"Recall    after  : {rec  * 100:.2f}%")
    print(f"F1 Score  after  : {f1   * 100:.2f}%")
    print(f"Time             : {elapsed:.1f}s")
    print("=" * 55)
    print("\n✔  Vision Mamba files were NOT modified.")
    print(f"✔  Best model → {BEST_MODEL_PATH}")
    print(f"✔  Synced     → {FALLBACK_MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int, default=FINETUNE_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    a = p.parse_args()
    finetune(epochs=a.epochs, batch_size=a.batch_size)
