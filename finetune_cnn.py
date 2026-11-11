# ==============================================================================
# CNN TARGETED FINE-TUNING — ROOT LAUNCHER
# Run from the project root:  python finetune_cnn.py
# ==============================================================================
import sys
import os

# Ensure src/ is on the path so local imports work
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from finetune_cnn_weak_classes import finetune

if __name__ == "__main__":
    finetune(epochs=5, batch_size=32)
