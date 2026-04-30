"""
SVM Cat vs Dog Classifier
==========================
Folder structure expected:
    Animal/
    ├── training_set/
    │   ├── cats/
    │   └── dogs/
    └── test_set/
        ├── cats/
        └── dogs/
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from joblib import dump
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION — edit BASE_DIR if needed
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))  # Animal/ folder
TRAIN_DIR  = os.path.join(BASE_DIR, "training_set")
TEST_DIR   = os.path.join(BASE_DIR, "test_set")
MODEL_PATH = os.path.join(BASE_DIR, "svm_cat_dog.joblib")

IMG_SIZE             = (64, 64)
HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (8, 8)
HOG_CELLS_PER_BLOCK  = (2, 2)
MAX_SAMPLES_PER_CLASS = 2000   # increase for better accuracy (slower)
RANDOM_STATE         = 42

# ─────────────────────────────────────────────
# 1. LOAD IMAGES
# ─────────────────────────────────────────────
def load_images_from_folder(folder: str, label: int, max_samples: int):
    """Load images from a single class folder."""
    X, y = [], []
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    files = files[:max_samples]

    for fname in tqdm(files, desc=f"  Loading {os.path.basename(folder)}", leave=False):
        path = os.path.join(folder, fname)
        img  = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X.append(img)
        y.append(label)

    return X, y


def load_dataset(root_dir: str):
    """Load cats (label=0) and dogs (label=1) from training_set or test_set."""
    cats_dir = os.path.join(root_dir, "cats")
    dogs_dir = os.path.join(root_dir, "dogs")

    for d in [cats_dir, dogs_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Folder not found: {d}")

    print(f"\n[INFO] Loading from: {root_dir}")
    Xc, yc = load_images_from_folder(cats_dir, label=0, max_samples=MAX_SAMPLES_PER_CLASS)
    Xd, yd = load_images_from_folder(dogs_dir, label=1, max_samples=MAX_SAMPLES_PER_CLASS)

    X = np.array(Xc + Xd)
    y = np.array(yc + yd)
    print(f"[INFO] Loaded {len(Xc)} cats + {len(Xd)} dogs = {len(X)} total")
    return X, y


# ─────────────────────────────────────────────
# 2. HOG FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_hog_features(images: np.ndarray) -> np.ndarray:
    features = []
    for img in tqdm(images, desc="Extracting HOG features"):
        feat = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        features.append(feat)
    return np.array(features)


# ─────────────────────────────────────────────
# 3. VISUALIZE HOG
# ─────────────────────────────────────────────
def visualize_hog(images: np.ndarray, labels: np.ndarray, n: int = 4):
    label_names = {0: 'Cat', 1: 'Dog'}
    fig, axes = plt.subplots(n, 2, figsize=(7, n * 3))
    fig.suptitle('HOG Feature Visualization', fontsize=13, y=1.01)

    for i in range(n):
        _, hog_img = hog(
            images[i],
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm='L2-Hys',
            visualize=True,
            feature_vector=True
        )
        axes[i, 0].imshow(images[i], cmap='gray')
        axes[i, 0].set_title(f'Original ({label_names[labels[i]]})', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(hog_img, cmap='magma')
        axes[i, 1].set_title('HOG Features', fontsize=10)
        axes[i, 1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, 'hog_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] HOG visualization saved → {save_path}")


# ─────────────────────────────────────────────
# 4. TRAIN SVM
# ─────────────────────────────────────────────
def train_svm(X_train: np.ndarray, y_train: np.ndarray):
    print("\n[INFO] Training SVM pipeline (StandardScaler + RBF SVC)...")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                       probability=True, random_state=RANDOM_STATE))
    ])

    # 5-fold cross-validation on training data
    print("[INFO] Running 5-fold cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=5, scoring='accuracy', n_jobs=-1)
    print(f"[INFO] CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Final fit on full training data
    pipeline.fit(X_train, y_train)
    print("[INFO] Training complete!")
    return pipeline


# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print("        MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

    # Confusion matrix plot
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Cat', 'Dog'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix — SVM Cat vs Dog', fontsize=12, pad=12)
    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Confusion matrix saved → {save_path}")

    # Precision / Recall / F1 bar chart
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred, average=None)
    recall    = recall_score(y_test, y_pred, average=None)
    f1        = f1_score(y_test, y_pred, average=None)

    classes = ['Cat', 'Dog']
    x       = np.arange(len(classes))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width, precision * 100, width, label='Precision', color='#1D9E75', edgecolor='white')
    bars2 = ax.bar(x,         recall    * 100, width, label='Recall',    color='#185FA5', edgecolor='white')
    bars3 = ax.bar(x + width, f1        * 100, width, label='F1-Score',  color='#7F77DD', edgecolor='white')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylim(60, 85)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Precision / Recall / F1-Score — Cat vs Dog', fontsize=12, pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    pr_path = os.path.join(BASE_DIR, 'precision_recall_chart.png')
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Precision/Recall chart saved → {pr_path}")

    return acc


# ─────────────────────────────────────────────
# 6. PREDICT A SINGLE IMAGE
# ─────────────────────────────────────────────
def predict_image(model, img_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read: {img_path}")
        return

    gray     = cv2.resize(img, IMG_SIZE)
    gray     = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    features = extract_hog_features([gray])

    pred  = model.predict(features)[0]
    prob  = model.predict_proba(features)[0]
    label = 'Dog' if pred == 1 else 'Cat'
    conf  = prob[pred] * 100

    print(f"\n[RESULT] → {label} ({conf:.1f}% confidence)")

    rgb = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(3, 3))
    plt.imshow(rgb)
    plt.title(f'{label}  ({conf:.1f}%)', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("   SVM Cat vs Dog Classifier")
    print("=" * 50)

    # ── Load training data ──────────────────────
    X_train_raw, y_train = load_dataset(TRAIN_DIR)

    # ── Load test data ──────────────────────────
    X_test_raw, y_test = load_dataset(TEST_DIR)

    # ── Extract HOG features ────────────────────
    print("\n[INFO] Extracting HOG features — training set...")
    X_train = extract_hog_features(X_train_raw)

    print("[INFO] Extracting HOG features — test set...")
    X_test  = extract_hog_features(X_test_raw)

    print(f"[INFO] HOG feature vector size: {X_train.shape[1]}")

    # ── Visualize HOG on a few samples ──────────
    visualize_hog(X_train_raw[:4], y_train[:4])

    # ── Train SVM ───────────────────────────────
    model = train_svm(X_train, y_train)

    # ── Evaluate on test set ────────────────────
    evaluate_model(model, X_test, y_test)

    # ── Save model 
    dump(model, MODEL_PATH)
    print(f"\n[INFO] Model saved → {MODEL_PATH}")

    print("\n[DONE] All steps complete!")


if __name__ == "__main__":
    main()