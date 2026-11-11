# ==============================================================================
# CNN RETRAINING CONTINUATION SCRIPT (10 ADDITIONAL EPOCHS)
# Preserves existing checkpoint weights and recovers balanced 38-class representation
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
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Setup Paths
BASE_DIR = r"c:\Users\Gango\Desktop\crop disease project\Crop_Disease_Project"
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATASET_PATH = os.path.join(BASE_DIR, "Dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "Saved_Models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_cnn_model.h5")
FALLBACK_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "cnn_model.h5")
CLASS_INDEX_PATH = os.path.join(MODELS_DIR, "class_indices.json")

from model import create_model


class SaveEveryEpochCallback(Callback):
    """Saves a checkpoint every single epoch, and saves the best model based on val_accuracy."""
    def __init__(self, checkpoints_dir, best_model_path, fallback_model_path):
        super().__init__()
        self.checkpoints_dir = checkpoints_dir
        self.best_model_path = best_model_path
        self.fallback_model_path = fallback_model_path
        self.best_val_acc = -1.0
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = float(logs.get('val_accuracy', 0.0))
        val_loss = float(logs.get('val_loss', 0.0))
        train_acc = float(logs.get('accuracy', 0.0))
        train_loss = float(logs.get('loss', 0.0))
        
        epoch_num = epoch + 1
        checkpoint_filename = f"cnn_epoch_{epoch_num:02d}_valacc_{val_acc:.4f}.h5"
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_filename)
        
        # Save per-epoch checkpoint
        self.model.save(checkpoint_path)
        print(f"\n[Checkpoint] Saved Epoch {epoch_num} checkpoint -> {checkpoint_filename} (Val Acc: {val_acc*100:.2f}%)")

        # Track and save best model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch_num
            self.model.save(self.best_model_path)
            self.model.save(self.fallback_model_path)
            print(f"[Best Model] New best val_accuracy: {val_acc*100:.2f}% (Epoch {epoch_num}). Updated {self.best_model_path} and {self.fallback_model_path}")


def load_dataset_samples(dataset_path, class_to_idx, samples_per_class=150, img_size=128, val_split=0.2, seed=42):
    np.random.seed(seed)
    train_x, train_y = [], []
    val_x, val_y = [], []

    classes = sorted(class_to_idx.keys())
    print(f"[Data] Loading stratified balanced samples ({samples_per_class}/class) across {len(classes)} classes...")

    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        if not os.path.isdir(c_dir):
            continue
        all_f = sorted([f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        np.random.shuffle(all_f)

        selected = all_f[:samples_per_class] if (samples_per_class and samples_per_class > 0) else all_f
        num_val = max(1, int(len(selected) * val_split))
        val_names = selected[:num_val]
        train_names = selected[num_val:]
        target = class_to_idx[c]

        for fname in train_names:
            fpath = os.path.join(c_dir, fname)
            try:
                img = tf.keras.preprocessing.image.load_img(fpath, target_size=(img_size, img_size))
                train_x.append(tf.keras.preprocessing.image.img_to_array(img) / 255.0)
                train_y.append(target)
            except Exception:
                pass

        for fname in val_names:
            fpath = os.path.join(c_dir, fname)
            try:
                img = tf.keras.preprocessing.image.load_img(fpath, target_size=(img_size, img_size))
                val_x.append(tf.keras.preprocessing.image.img_to_array(img) / 255.0)
                val_y.append(target)
            except Exception:
                pass

    return np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.int32), \
           np.array(val_x, dtype=np.float32), np.array(val_y, dtype=np.int32)


def run_training_continuation(epochs=10, batch_size=32, samples_per_class=150, lr=3e-4):
    start_time = time.time()
    print("=" * 70)
    print(f"CONTINUING CNN TRAINING FOR {epochs} ADDITIONAL EPOCHS")
    print("Preserving existing weights, saving checkpoints each epoch")
    print("=" * 70)

    # Load class mappings
    with open(CLASS_INDEX_PATH, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)

    # 1. Load balanced dataset
    X_train, y_train, X_val, y_val = load_dataset_samples(
        DATASET_PATH,
        class_to_idx,
        samples_per_class=samples_per_class,
        img_size=128,
        val_split=0.2,
        seed=42
    )
    print(f"[Data Loaded] Train samples: {len(X_train)} | Val samples: {len(X_val)} | Total: {len(X_train)+len(X_val)}")

    # 2. Instantiate and load weights from existing checkpoint
    print(f"\n[Model] Loading existing weights from: {BEST_MODEL_PATH}")
    model = create_model(num_classes)
    if os.path.exists(BEST_MODEL_PATH):
        try:
            model.load_weights(BEST_MODEL_PATH)
            print("[Model] Existing weights loaded successfully from checkpoint!")
        except Exception as e:
            print(f"[Model] Note: loading via load_model due to: {e}")
            model = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
    else:
        print("[Model] Fallback to Saved_Models/cnn_model.h5")
        model.load_weights(FALLBACK_MODEL_PATH)

    # Recompile with learning rate and Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Evaluate baseline before additional epochs
    init_val_loss, init_val_acc = model.evaluate(X_val, y_val, batch_size=64, verbose=0)
    print(f"[Baseline Before Retraining] Val Loss: {init_val_loss:.4f} | Val Accuracy: {init_val_acc*100:.2f}%")

    # 3. Data Augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_flow = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)

    # 4. Callbacks: Save every epoch + Reduce LR on plateau
    save_cb = SaveEveryEpochCallback(CHECKPOINTS_DIR, BEST_MODEL_PATH, FALLBACK_MODEL_PATH)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )

    # 5. Execute 10 epochs
    history = model.fit(
        train_flow,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[save_cb, reduce_lr]
    )

    training_time = time.time() - start_time
    print(f"\n[Finished] 10 Epochs completed in {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"[Best Epoch] Best Val Accuracy: {save_cb.best_val_acc*100:.2f}% at Epoch {save_cb.best_epoch}")

    # 6. Load the best checkpoint weights for final evaluation
    model.load_weights(BEST_MODEL_PATH)

    # 7. Generate Training & Validation Curves
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, epochs + 1), history.history['accuracy'], 'b-o', label='Training Accuracy')
    plt.plot(range(1, epochs + 1), history.history['val_accuracy'], 'g-s', label='Validation Accuracy')
    plt.title('CNN Retraining: Accuracy Curves (10 Additional Epochs)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    acc_curve_path = os.path.join(RESULTS_DIR, "cnn_accuracy_curve.png")
    plt.savefig(acc_curve_path, dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, epochs + 1), history.history['loss'], 'b-o', label='Training Loss')
    plt.plot(range(1, epochs + 1), history.history['val_loss'], 'r-s', label='Validation Loss')
    plt.title('CNN Retraining: Loss Curves (10 Additional Epochs)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Categorical Cross-Entropy Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_curve_path = os.path.join(RESULTS_DIR, "cnn_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()

    # 8. Complete Evaluation on Validation Set
    val_probs = model.predict(X_val, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(val_probs, axis=1)
    val_confs = np.max(val_probs, axis=1) * 100.0

    acc = float(accuracy_score(y_val, y_pred))
    prec = float(precision_score(y_val, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_val, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_val, y_pred, average='weighted', zero_division=0))
    support = int(len(y_val))

    correct_mask = (y_pred == y_val)
    incorrect_mask = (y_pred != y_val)
    avg_conf_corr = float(np.mean(val_confs[correct_mask])) if np.any(correct_mask) else 0.0
    avg_conf_incorr = float(np.mean(val_confs[incorrect_mask])) if np.any(incorrect_mask) else 0.0

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Retrained CNN Confusion Matrix (Accuracy: {acc*100:.2f}%)', fontsize=16)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "cnn_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Classification Report
    cls_rep = classification_report(y_val, y_pred, target_names=class_names, zero_division=0)
    rep_path = os.path.join(RESULTS_DIR, "classification_report_cnn.txt")
    with open(rep_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"RETRAINED CNN CLASSIFICATION REPORT (10 Additional Epochs)\n")
        f.write("=" * 60 + "\n\n")
        f.write(cls_rep)
        f.write(f"\nTraining Time: {training_time:.2f}s | Best Val Accuracy: {save_cb.best_val_acc*100:.2f}%\n")

    # Metrics JSON
    metrics = {
        "model": "CNN",
        "epochs_trained": epochs,
        "training_time_seconds": round(training_time, 2),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "support": support,
        "avg_confidence_correct": round(avg_conf_corr, 2),
        "avg_confidence_incorrect": round(avg_conf_incorr, 2)
    }
    metrics_path = os.path.join(RESULTS_DIR, "cnn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Class-wise accuracy
    class_acc = {}
    for i in range(num_classes):
        tot = np.sum(y_val == i)
        corr = np.sum((y_val == i) & (y_pred == i))
        class_acc[class_names[i]] = float(corr / tot) if tot > 0 else 0.0

    with open(os.path.join(RESULTS_DIR, "cnn_class_accuracy.json"), "w") as f:
        json.dump(class_acc, f, indent=4)

    print("\n" + "=" * 50)
    print("RETRAINED CNN FINAL RESULTS")
    print("=" * 50)
    print(f"Accuracy:                 {acc * 100:.2f}%")
    print(f"Precision (Weighted):     {prec * 100:.2f}%")
    print(f"Recall (Weighted):        {rec * 100:.2f}%")
    print(f"F1 Score (Weighted):      {f1 * 100:.2f}%")
    print(f"Avg Confidence (Correct): {avg_conf_corr:.2f}%")
    print(f"Avg Confidence (Incorr):  {avg_conf_incorr:.2f}%")
    print(f"Checkpoints Saved:        10 in models/checkpoints/")
    print("=" * 50)

    return metrics, class_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples_per_class", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    run_training_continuation(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,
        lr=args.lr
    )
