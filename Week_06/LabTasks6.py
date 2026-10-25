# ---------------Activity 1---------------
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
path = kagglehub.dataset_download("zalando-research/fashionmnist")
data = pd.read_csv(os.path.join(path, "fashion-mnist_train.csv"))
print("Dataset shape:", data.shape)
print("\nFirst ten rows:")
print(data.head(10))
print("\nClass distribution:")
print(data["label"].value_counts().sort_index())
X = data.drop(columns=["label"])
y = data["label"]
X_norm = X / 255.0
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Mean after scaling:", round(X_scaled.mean(), 4))
print("Std after scaling:", round(X_scaled.std(), 4))
pca_fixed = PCA(n_components=50)
X_fixed = pca_fixed.fit_transform(X_scaled)
pca_var = PCA(n_components=0.95)
X_var = pca_var.fit_transform(X_scaled)
pca_all = PCA(n_components=None)
X_all = pca_all.fit_transform(X_scaled)
print("Method 1: Fixed number of components")
print("Original features:", X.shape[1])
print("Retained components:", pca_fixed.n_components_)
print("Cumulative explained variance:",
      np.sum(pca_fixed.explained_variance_ratio_))
print("\nMethod 2: Percentage of variance")
print("Original features:", X.shape[1])
print("Retained components:", pca_var.n_components_)
print("Cumulative explained variance:",
      np.sum(pca_var.explained_variance_ratio_))
print("\nMethod 3: All components")
print("Original features:", X.shape[1])
print("Retained components:", pca_all.n_components_)
print("Cumulative explained variance:",
      np.sum(pca_all.explained_variance_ratio_))
cumsum = np.cumsum(pca_all.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cumsum) + 1),
    cumsum
)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.grid()
plt.show()
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=y,
    cmap='jet',
    edgecolors='none',
    alpha=0.7
)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization of Fashion MNIST Dataset')
plt.colorbar(scatter, label='Clothing Class')
plt.grid(alpha=0.2)
plt.show()
comparison = pd.DataFrame({
    "Method": ["Fixed (50)", "Variance (0.95)", "All components"],
    "Original Features": [X.shape[1]] * 3,
    "Retained Components": [
        pca_fixed.n_components_,
        pca_var.n_components_,
        pca_all.n_components_
    ],
    "Cumulative Explained Variance": [
        np.sum(pca_fixed.explained_variance_ratio_),
        np.sum(pca_var.explained_variance_ratio_),
        np.sum(pca_all.explained_variance_ratio_)
    ]
})
print(comparison)




# ---------------Activity 2---------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
path = kagglehub.dataset_download("redwankarimsony/heart-disease-data")
data = pd.read_csv(os.path.join(path, "heart_disease_uci.csv"))
print("Dataset shape:", data.shape)
print("\nFirst ten rows:")
print(data.head(10))
print("\nMissing values:")
print(data.isnull().sum())
data["target"] = (data["num"] > 0).astype(int)
print("\nTarget class distribution:")
print(data["target"].value_counts())
data = data.drop(columns=["id", "dataset", "num"])
num_cols = data.select_dtypes(include=np.number).columns.drop("target")
cat_cols = data.select_dtypes(exclude=np.number).columns
for col in num_cols:
    data[col] = data[col].fillna(data[col].median())
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])
data = pd.get_dummies(data, columns=cat_cols, drop_first=True, dtype=int)
print("\nMissing values after cleaning:", data.isnull().sum().sum())
print("Shape after encoding:", data.shape)
X = data.drop(columns=["target"])
y = data["target"]
print("Features shape:", X.shape)
print("Target shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
rf_basic = RandomForestClassifier()
rf_basic.fit(X_train, y_train)
y_pred_basic = rf_basic.predict(X_test)
basic_acc = accuracy_score(y_test, y_pred_basic)
print("\nBasic Random Forest Test Accuracy:", round(basic_acc, 4))
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy"
)
grid_search.fit(X_train, y_train)
print("\nBest hyperparameters:", grid_search.best_params_)
print("Best cross-validation accuracy:",
      round(grid_search.best_score_, 4))
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nTest Set Results")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-score:", round(f1_score(y_test, y_pred), 4))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.head(15).sort_values().plot(kind="barh")
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
print("\nThree most influential features:")
print(importances.head(3))




# ---------------Activity 3---------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
data = pd.read_csv("train.csv")
print("Dataset shape:", data.shape)
print("\nFirst ten rows:")
print(data.head(10))
missing = data.isnull().sum()
print("\nMissing values:")
print(missing[missing > 0].sort_values(ascending=False))
print("\nSalePrice summary statistics:")
print(data["SalePrice"].describe())
X = data.drop(columns=["SalePrice", "Id"])
y = data["SalePrice"]
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
for col in cat_cols:
    X[col] = X[col].fillna("None")
X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
print("\nMissing values after cleaning:", X.isnull().sum().sum())
print("Shape after encoding:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
rf_basic = RandomForestRegressor()
rf_basic.fit(X_train, y_train)
y_pred_basic = rf_basic.predict(X_test)
basic_mae = mean_absolute_error(y_test, y_pred_basic)
basic_mse = mean_squared_error(y_test, y_pred_basic)
basic_rmse = np.sqrt(basic_mse)
basic_r2 = r2_score(y_test, y_pred_basic)
print("\nBasic Random Forest Results")
print("MAE:", round(basic_mae, 2))
print("MSE:", round(basic_mse, 2))
print("RMSE:", round(basic_rmse, 2))
print("R2 Score:", round(basic_r2, 4))
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": [0.7, 1.0]
}
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)
grid_search.fit(X_train, y_train)
best_mse_cv = -grid_search.best_score_
best_rmse_cv = np.sqrt(best_mse_cv)
print("\nBest hyperparameters:", grid_search.best_params_)
print("Best cross-validation score (neg MSE):",
      round(grid_search.best_score_, 2))
print("Best cross-validation MSE:", round(best_mse_cv, 2))
print("Best cross-validation RMSE:", round(best_rmse_cv, 2))
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
opt_mae = mean_absolute_error(y_test, y_pred)
opt_mse = mean_squared_error(y_test, y_pred)
opt_rmse = np.sqrt(opt_mse)
opt_r2 = r2_score(y_test, y_pred)
print("\nOptimized Random Forest Results")
print("MAE:", round(opt_mae, 2))
print("MSE:", round(opt_mse, 2))
print("RMSE:", round(opt_rmse, 2))
print("R2 Score:", round(opt_r2, 4))
comparison = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2"],
    "Default Model": [basic_mae, basic_mse, basic_rmse, basic_r2],
    "Optimized Model": [opt_mae, opt_mse, opt_rmse, opt_r2]
})
print("\nModel Comparison:")
print(comparison)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(alpha=0.2)
plt.show()
importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
top5 = importances.head(5)
plt.figure(figsize=(8, 5))
top5.sort_values().plot(kind="barh")
plt.title("Top 5 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
print("\nFive most influential features:")
print(top5)
