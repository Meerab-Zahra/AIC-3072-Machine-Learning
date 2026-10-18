# Activity 2: SVM for brain tumor detection
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
DATASET_PATH = None  
if DATASET_PATH is None:
    import kagglehub
    DATASET_PATH = kagglehub.dataset_download(
        "navoneel/brain-mri-images-for-brain-tumor-detection"
    )
print("Dataset path:", DATASET_PATH)
def find_class_folder(root, name):
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() == name:
                return os.path.join(dirpath, d)
    raise FileNotFoundError(f"Could not find a '{name}' folder inside {root}")
IMG_SIZE = 64
CLASS_NAMES = ["No Tumor", "Tumor"]  
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
X, y = [], []
for label, folder_name in enumerate(["no", "yes"]):
    folder = find_class_folder(DATASET_PATH, folder_name)
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(VALID_EXT):
            continue
        img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)  # grayscale
        if img is None:  
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  
        X.append(img.flatten())                    
        y.append(label)
X = np.array(X, dtype=np.float32)
y = np.array(y)
print(f"Total images : {len(X)}")
print(f"Feature shape: {X.shape}")
print(f"Class counts : No Tumor = {np.sum(y == 0)}, Tumor = {np.sum(y == 1)}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM (RBF)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
num_show = 10
rng = np.random.default_rng(42)
indices = rng.choice(len(X_test), size=min(num_show, len(X_test)), replace=False)
plt.figure(figsize=(15, 6))
for i, idx in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
    actual = CLASS_NAMES[y_test[idx]]
    predicted = CLASS_NAMES[y_pred[idx]]
    color = "green" if actual == predicted else "red"
    plt.title(f"Actual: {actual}\nPred: {predicted}", color=color, fontsize=10)
    plt.axis("off")
plt.suptitle("Test Images: Predicted vs Actual (green = correct, red = wrong)")
plt.tight_layout()
plt.savefig("test_predictions.png", dpi=150)
plt.show()

# Activity 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
CSV_PATH = "Mall_Customers.csv"   
RANDOM_STATE = 42
K = 5                        
INCOME = "Annual Income (k$)"
SCORE = "Spending Score (1-100)"
df = pd.read_csv(CSV_PATH)
print("First 10 rows:\n", df.head(10), "\n")
print("Shape:", df.shape, "\n")
df.info()
print("\nSummary statistics:\n", df.describe(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")
X = df[[INCOME, SCORE]].values
def scatter_clusters(ax, data, labels, title):
    for lab in sorted(set(labels)):
        mask = labels == lab
        if lab == -1:
            ax.scatter(data[mask, 0], data[mask, 1], c="black", marker="x", s=40, label="Noise")
        else:
            ax.scatter(data[mask, 0], data[mask, 1], s=40, label=f"Cluster {lab}")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title(title)
    ax.legend(fontsize=8)
kmeans_raw = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
labels_km_raw = kmeans_raw.fit_predict(X)
dbscan = DBSCAN(eps=5, min_samples=5)   
labels_db = dbscan.fit_predict(X)
n_db_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = int(np.sum(labels_db == -1))
print(f"K-Means (raw)  : {K} clusters, silhouette = {silhouette_score(X, labels_km_raw):.3f}")
print(f"DBSCAN (raw)   : {n_db_clusters} clusters, {n_noise} noise points")
if n_db_clusters > 1:
    mask = labels_db != -1
    print(f"silhouette (non-noise) = {silhouette_score(X[mask], labels_db[mask]):.3f}")
print()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
scatter_clusters(axes[0], X, labels_km_raw, f"K-Means (k={K}), unscaled")
axes[0].scatter(*kmeans_raw.cluster_centers_.T, c="red", marker="*", s=250,
                edgecolors="black", label="Centroids")
axes[0].legend(fontsize=8)
scatter_clusters(axes[1], X, labels_db, "DBSCAN (eps=5, min_samples=5), unscaled")
plt.tight_layout()
plt.savefig("01_kmeans_vs_dbscan.png", dpi=150)
plt.show()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_scaled = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
labels_km_scaled = kmeans_scaled.fit_predict(X_scaled)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
scatter_clusters(axes[0], X, labels_km_raw, "K-Means, unscaled")
scatter_clusters(axes[1], X, labels_km_scaled, "K-Means, scaled (shown in original units)")
plt.tight_layout()
plt.savefig("02_kmeans_scaled_vs_unscaled.png", dpi=150)
plt.show()
ari_scale = adjusted_rand_score(labels_km_raw, labels_km_scaled)
sil_raw = silhouette_score(X, labels_km_raw)
sil_scaled = silhouette_score(X_scaled, labels_km_scaled)
n_changed = int(np.sum(
    pd.crosstab(labels_km_raw, labels_km_scaled).values.sum()
    - pd.crosstab(labels_km_raw, labels_km_scaled).values.max(axis=1).sum()
))

print("=== Did scaling change the clusters? ===")
print(f"Adjusted Rand Index (raw vs scaled): {ari_scale:.3f}")
print(f"Silhouette  raw: {sil_raw:.3f} | scaled: {sil_scaled:.3f}")
print(f"Approx. customers whose group changed: {n_changed} of {len(df)}")
print("Contingency table (rows = raw clusters, columns = scaled clusters):")
print(pd.crosstab(labels_km_raw, labels_km_scaled), "\n")
print("Why: Income (about 15-137) and Spending Score (1-99) already have similar ranges, so")
print("StandardScaler barely changes relative distances. Scaling matters a lot when features")
print("have very different ranges (for example Income in thousands vs Age in tens).\n")
k_range = range(1, 11)
inertias, silhouettes = [], []
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_) if k > 1 else np.nan)
best_k_sil = list(k_range)[int(np.nanargmax(silhouettes))]
print(f"Best k by silhouette score: {best_k_sil}")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(list(k_range), inertias, "o-")
axes[0].axvline(K, color="red", linestyle="--", label=f"k = {K}")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_title("Elbow Method")
axes[0].legend()
axes[1].plot(list(k_range), silhouettes, "o-", color="green")
axes[1].axvline(best_k_sil, color="red", linestyle="--", label=f"best k = {best_k_sil}")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette Check")
axes[1].legend()
plt.tight_layout()
plt.savefig("03_elbow_method.png", dpi=150)
plt.show()
df["Cluster"] = labels_km_scaled
summary = df.groupby("Cluster").agg(
    Customers=("Cluster", "size"),
    Avg_Age=("Age", "mean") if "Age" in df.columns else ("Cluster", "size"),
    Avg_Income=(INCOME, "mean"),
    Avg_Score=(SCORE, "mean"),
).round(1)
def describe_cluster(income, score):
    inc = "Low" if income < 40 else ("High" if income > 70 else "Average")
    spd = "Low" if score < 40 else ("High" if score > 60 else "Average")
    names = {
        ("Low", "Low"): "Low income, low spending: budget-conscious / sensible",
        ("Low", "High"): "Low income, high spending: impulsive / careless spenders",
        ("High", "Low"): "High income, low spending: careful, under-targeted by the mall",
        ("High", "High"): "High income, high spending: premium target customers",
    }
    return names.get((inc, spd), "Average income, average spending: standard shoppers")
summary["Interpretation"] = [
    describe_cluster(r.Avg_Income, r.Avg_Score) for r in summary.itertuples()
]
pd.set_option("display.width", 250, "display.max_colwidth", 80, "display.max_columns", None)
print("=== Cluster interpretation ===")
print(summary, "\n")
rng = np.random.default_rng(RANDOM_STATE)
df["LuckyNumber"] = rng.integers(0, 101, size=len(df))
X_lucky = df[[INCOME, SCORE, "LuckyNumber"]].values
X_lucky_scaled = StandardScaler().fit_transform(X_lucky)
km_lucky_raw = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(X_lucky)
km_lucky_scaled = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE).fit(X_lucky_scaled)
print("=== Effect of the fake LuckyNumber feature ===")
print(f"{'Model':<34}{'ARI vs clean':>14}{'Silhouette (real features)':>30}")
print(f"{'Baseline (Income + Score)':<34}{1.0:>14.3f}{sil_scaled:>30.3f}")
for name, lab in [("+ LuckyNumber, unscaled", km_lucky_raw.labels_),
                  ("+ LuckyNumber, scaled", km_lucky_scaled.labels_)]:
    print(f"{name:<34}{adjusted_rand_score(labels_km_scaled, lab):>14.3f}"
          f"{silhouette_score(X_scaled, lab):>30.3f}")
print()
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
scatter_clusters(axes[0], X, labels_km_scaled, "Baseline (Income + Score)")
scatter_clusters(axes[1], X, km_lucky_raw.labels_, "With LuckyNumber (unscaled)")
scatter_clusters(axes[2], X, km_lucky_scaled.labels_, "With LuckyNumber (scaled)")
plt.tight_layout()
plt.savefig("04_luckynumber_effect.png", dpi=150)
plt.show()
print("Explanation:")
print("- LuckyNumber is pure random noise: it has no relationship to income or spending.")
print("- K-Means uses Euclidean distance over ALL features, so the noise feature adds random")
print("  distance between customers who are otherwise similar.")
print("- Its range (0-100) is as large as the real features, so it carries about as much")
print("  weight as them. Some clusters get split or merged by luck instead of behaviour.")
print("- In the Income vs Score plot the clusters look messier and overlap, ARI drops below")
print("  1, and the silhouette on the real features gets worse.")
print("- Scaling does NOT fix this. StandardScaler gives the noise feature equal weight too.")
print("- Lesson: irrelevant features hurt distance-based clustering. Use feature selection")
print("  or domain knowledge to remove them before clustering.")

