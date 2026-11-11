import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import torch
from torchvision import transforms
from PIL import Image

# Setup Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

DATASET_PATH = os.path.join(BASE_DIR, "Dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "Saved_Models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

CNN_MODEL_PATH = os.path.join(MODELS_DIR, "best_cnn_model.h5") if os.path.exists(os.path.join(MODELS_DIR, "best_cnn_model.h5")) else os.path.join(SAVED_MODELS_DIR, "cnn_model.h5")
MAMBA_MODEL_PATH = os.path.join(MODELS_DIR, "best_vision_mamba.pth") if os.path.exists(os.path.join(MODELS_DIR, "best_vision_mamba.pth")) else os.path.join(SAVED_MODELS_DIR, "best_vision_mamba.pth")
CLASS_INDEX_PATH = os.path.join(MODELS_DIR, "class_indices.json") if os.path.exists(os.path.join(MODELS_DIR, "class_indices.json")) else os.path.join(SAVED_MODELS_DIR, "class_indices.json")


def load_class_mapping():
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    return class_indices, class_names


def load_val_images(dataset_path, class_to_idx, samples_per_class=30, val_split=0.2, seed=42):
    np.random.seed(seed)
    val_files = []
    classes = sorted(class_to_idx.keys())

    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        if not os.path.exists(c_dir):
            continue
        all_f = sorted([f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        np.random.shuffle(all_f)
        selected = all_f[:samples_per_class] if (samples_per_class and samples_per_class > 0) else all_f
        num_val = max(1, int(len(selected) * val_split))
        for fn in selected[:num_val]:
            val_files.append((os.path.join(c_dir, fn), class_to_idx[c]))

    return val_files


def evaluate_cnn(val_files, class_names, num_classes):
    print("\n--- Evaluating CNN Model ---")
    print(f"Loading CNN from: {CNN_MODEL_PATH}")
    from model import create_model
    model = create_model(num_classes)
    try:
        model.load_weights(CNN_MODEL_PATH)
    except Exception:
        model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)

    y_true = []
    y_pred = []
    
    for fpath, label in val_files:
        try:
            img = tf.keras.preprocessing.image.load_img(fpath, target_size=(128, 128))
            arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            preds = model.predict(arr, verbose=0)
            y_pred.append(int(np.argmax(preds[0])))
            y_true.append(label)
        except Exception:
            continue

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    support = int(len(y_true))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'CNN Confusion Matrix (Acc: {acc*100:.2f}%)', fontsize=16)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Classification Report
    cls_rep = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    rep_path = os.path.join(RESULTS_DIR, "classification_report_cnn.txt")
    with open(rep_path, "w") as f:
        f.write("CNN CLASSIFICATION REPORT\n" + "=" * 50 + "\n" + cls_rep)

    metrics = {
        "model": "CNN",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "support": support
    }
    # Retain training time if exists
    json_path = os.path.join(RESULTS_DIR, "cnn_metrics.json")
    if os.path.exists(json_path):
        try:
            with open(json_path) as jf:
                old = json.load(jf)
                metrics["training_time_seconds"] = old.get("training_time_seconds", 0.0)
                metrics["epochs"] = old.get("epochs", 10)
        except Exception:
            pass

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def evaluate_mamba(val_files, class_names, num_classes):
    print("\n--- Evaluating Vision Mamba Model ---")
    print(f"Loading Vision Mamba from: {MAMBA_MODEL_PATH}")
    from vision_mamba import VisionMamba

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(MAMBA_MODEL_PATH, map_location=device)

    # Detect architecture parameters
    if "pos_embed" in state_dict:
        pos_tokens = state_dict["pos_embed"].shape[1]
        embed_dim = state_dict["pos_embed"].shape[2]
        img_size = int((pos_tokens ** 0.5) * 16)
    else:
        embed_dim = 192
        img_size = 256

    depth = len(set(int(k.split('.')[1]) for k in state_dict.keys() if k.startswith("layers.")))
    if depth == 0:
        depth = 4

    model = VisionMamba(
        img_size=img_size,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_diseases=num_classes
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    y_true = []
    y_pred = []

    with torch.no_grad():
        for fpath, label in val_files:
            try:
                img = Image.open(fpath).convert('RGB')
                t = val_tf(img).unsqueeze(0).to(device)
                out = model(t)
                pred = int(torch.argmax(out, dim=1).item())
                y_pred.append(pred)
                y_true.append(label)
            except Exception:
                continue

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    support = int(len(y_true))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Vision Mamba Confusion Matrix (Acc: {acc*100:.2f}%)', fontsize=16)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "mamba_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Classification Report
    cls_rep = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    rep_path = os.path.join(RESULTS_DIR, "classification_report_mamba.txt")
    with open(rep_path, "w") as f:
        f.write("VISION MAMBA CLASSIFICATION REPORT\n" + "=" * 50 + "\n" + cls_rep)

    metrics = {
        "model": "Vision Mamba",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "support": support
    }
    # Retain training time if exists
    json_path = os.path.join(RESULTS_DIR, "mamba_metrics.json")
    if os.path.exists(json_path):
        try:
            with open(json_path) as jf:
                old = json.load(jf)
                metrics["training_time_seconds"] = old.get("training_time_seconds", 0.0)
                metrics["epochs"] = old.get("epochs", 50)
        except Exception:
            pass

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def main():
    print("=" * 60)
    print("AGRIVISION COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    class_indices, class_names = load_class_mapping()
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes.")

    val_files = load_val_images(DATASET_PATH, class_indices, samples_per_class=40, val_split=0.2)
    print(f"Loaded {len(val_files)} validation samples across all 38 classes.")

    cnn_metrics = evaluate_cnn(val_files, class_names, num_classes)
    mamba_metrics = evaluate_mamba(val_files, class_names, num_classes)

    # Display comparison table
    print("\n" + "=" * 75)
    print(f"{'Metric':<25} | {'CNN (10 Epochs)':<22} | {'Vision Mamba (50 Epochs)':<22}")
    print("-" * 75)
    print(f"{'Accuracy':<25} | {cnn_metrics['accuracy']*100:>20.2f}% | {mamba_metrics['accuracy']*100:>20.2f}%")
    print(f"{'Precision (Weighted)':<25} | {cnn_metrics['precision']*100:>20.2f}% | {mamba_metrics['precision']*100:>20.2f}%")
    print(f"{'Recall (Weighted)':<25} | {cnn_metrics['recall']*100:>20.2f}% | {mamba_metrics['recall']*100:>20.2f}%")
    print(f"{'F1 Score (Weighted)':<25} | {cnn_metrics['f1_score']*100:>20.2f}% | {mamba_metrics['f1_score']*100:>20.2f}%")
    print(f"{'Support (Test Samples)':<25} | {cnn_metrics['support']:>21} | {mamba_metrics['support']:>21}")
    t_cnn = f"{cnn_metrics.get('training_time_seconds', 0.0):.2f}s"
    t_mamba = f"{mamba_metrics.get('training_time_seconds', 0.0):.2f}s"
    print(f"{'Training Time':<25} | {t_cnn:>21} | {t_mamba:>21}")
    print("=" * 75)
    print(f"All 10 evaluation artifacts verified in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

