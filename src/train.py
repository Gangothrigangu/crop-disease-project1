# ==============================================================================
# CNN TRAINING SCRIPT  - PLANTVILLAGE DATASET
# ==============================================================================
import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from model import create_model

# ---------------------------------------------------
# PATHS & DIRECTORIES SETUP
# ---------------------------------------------------
DATASET_PATH = os.path.join(BASE_DIR, "Dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "Saved_Models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_cnn_model.h5")
FALLBACK_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "cnn_model.h5")
CLASS_INDEX_PATH = os.path.join(MODELS_DIR, "class_indices.json")
SAVED_CLASS_INDEX_PATH = os.path.join(SAVED_MODELS_DIR, "class_indices.json")


def load_dataset_samples(dataset_path, samples_per_class=150, img_size=128, val_split=0.2, seed=42):
    """
    Loads a balanced stratified dataset from Dataset directory across all 38 classes.
    If samples_per_class is None or <= 0, loads the entire dataset.
    """
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    np.random.seed(seed)
    train_x, train_y = [], []
    val_x, val_y = [], []

    print(f"[Data] Discovered {len(classes)} classes in {dataset_path}")
    print(f"[Data] Strategy: Stratified balanced sampling ({samples_per_class} per class)...")

    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        img_names = sorted([f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        np.random.shuffle(img_names)
        
        if samples_per_class and samples_per_class > 0:
            selected_names = img_names[:samples_per_class]
        else:
            selected_names = img_names

        num_val = max(1, int(len(selected_names) * val_split))
        val_imgs = selected_names[:num_val]
        train_imgs = selected_names[num_val:]

        target = class_to_idx[c]

        for fname in train_imgs:
            fpath = os.path.join(c_dir, fname)
            try:
                img = tf.keras.preprocessing.image.load_img(fpath, target_size=(img_size, img_size))
                arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                train_x.append(arr)
                train_y.append(target)
            except Exception:
                continue

        for fname in val_imgs:
            fpath = os.path.join(c_dir, fname)
            try:
                img = tf.keras.preprocessing.image.load_img(fpath, target_size=(img_size, img_size))
                arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                val_x.append(arr)
                val_y.append(target)
            except Exception:
                continue

    X_train = np.array(train_x, dtype=np.float32)
    y_train = np.array(train_y, dtype=np.int32)
    X_val = np.array(val_x, dtype=np.float32)
    y_val = np.array(val_y, dtype=np.int32)

    return X_train, y_train, X_val, y_val, class_to_idx


def train_cnn(epochs=10, batch_size=32, samples_per_class=150):
    start_time = time.time()
    print("=" * 60)
    print(f"Starting CNN Training for exactly {epochs} Epochs")
    print(f"Target: models/best_cnn_model.h5 | Results: results/")
    print("=" * 60)

    # 1. Load Data
    X_train, y_train, X_val, y_val, class_to_idx = load_dataset_samples(
        DATASET_PATH, 
        samples_per_class=samples_per_class, 
        img_size=128, 
        val_split=0.2
    )
    num_classes = len(class_to_idx)
    print(f"[Data] Training samples: {len(X_train)} | Validation samples: {len(X_val)} | Classes: {num_classes}")

    # Synchronize class indices
    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump(class_to_idx, f, indent=4)
    with open(SAVED_CLASS_INDEX_PATH, "w") as f:
        json.dump(class_to_idx, f, indent=4)
    print(f"[Setup] Class indices saved to {CLASS_INDEX_PATH} and {SAVED_CLASS_INDEX_PATH}")

    # 2. Data Augmentation Generator for Training
    train_gen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )
    train_flow = train_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)

    # 3. Instantiate & Compile Model
    model = create_model(num_classes)
    model.summary()

    # 4. Checkpoint Callbacks
    callbacks = [
        ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 5. Train for exactly 10 Epochs
    history = model.fit(
        train_flow,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks
    )

    training_time = time.time() - start_time
    print(f"\n[Training] CNN Training Finished in {training_time:.2f}s ({training_time/60:.2f} min)")

    # Ensure best weights are loaded into the model for evaluation
    if os.path.exists(BEST_MODEL_PATH):
        model.load_weights(BEST_MODEL_PATH)
        # Also sync to Saved_Models/cnn_model.h5
        model.save(FALLBACK_MODEL_PATH)
        print(f"[Model] Best model loaded and synchronized to {FALLBACK_MODEL_PATH}")

    # 6. Generate Training / Validation Curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), history.history['accuracy'], 'b-o', label='Training Accuracy')
    plt.plot(range(1, epochs + 1), history.history['val_accuracy'], 'g-s', label='Validation Accuracy')
    plt.title('CNN Model Training & Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    acc_curve_path = os.path.join(RESULTS_DIR, "cnn_accuracy_curve.png")
    plt.savefig(acc_curve_path, dpi=300)
    plt.close()
    print(f"[Results] Saved: {acc_curve_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), history.history['loss'], 'b-o', label='Training Loss')
    plt.plot(range(1, epochs + 1), history.history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('CNN Model Training & Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Categorical Cross-Entropy Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_curve_path = os.path.join(RESULTS_DIR, "cnn_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"[Results] Saved: {loss_curve_path}")

    # 7. Evaluate on Validation Set
    print("\n[Evaluation] Evaluating CNN on Validation Set...")
    y_pred_probs = model.predict(X_val, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = float(accuracy_score(y_val, y_pred))
    prec = float(precision_score(y_val, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_val, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_val, y_pred, average='weighted', zero_division=0))
    support = int(len(y_val))

    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]

    # 8. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'CNN Confusion Matrix (Accuracy: {acc*100:.2f}%)', fontsize=16)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Results] Saved: {cm_path}")

    # 9. Classification Report
    cls_report = classification_report(y_val, y_pred, target_names=class_names, zero_division=0)
    report_path = os.path.join(RESULTS_DIR, "classification_report_cnn.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"CNN CLASSIFICATION REPORT (Epochs: {epochs})\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_report)
        f.write(f"\nTraining Time: {training_time:.2f} seconds\n")
    print(f"[Results] Saved: {report_path}")

    # 10. Metrics JSON
    metrics = {
        "model": "CNN",
        "epochs": epochs,
        "training_time_seconds": round(training_time, 2),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "support": support
    }
    metrics_path = os.path.join(RESULTS_DIR, "cnn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Results] Saved: {metrics_path}")

    print("\n" + "=" * 50)
    print("CNN FINAL RESULTS")
    print("=" * 50)
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Precision:       {prec * 100:.2f}%")
    print(f"Recall:          {rec * 100:.2f}%")
    print(f"F1 Score:        {f1 * 100:.2f}%")
    print(f"Support:         {support}")
    print(f"Training Time:   {training_time:.2f}s")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on PlantVillage Dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default 5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default 32)")
    parser.add_argument("--samples_per_class", type=int, default=150, help="Samples per class (default 150)")
    args = parser.parse_args()

    train_cnn(epochs=args.epochs, batch_size=args.batch_size, samples_per_class=args.samples_per_class)