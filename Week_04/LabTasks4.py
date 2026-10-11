#-------------------Class Activiy-------------------
# Applying KNN to a classification problem
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
wine = load_wine()
X,y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
k_values = [1, 3, 5, 7, 9]
accuracy_scores = []
for k in k_values:  
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    accuracy_scores.append(accuracy)
plt.scatter(k_values, accuracy_scores)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. k Values")
plt.show()
k_values = range(1, 30)
accuracy_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
best_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print("The best value of k is:", best_k)




# Changing the values of p
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
p_values = [1, 2, 3, 4, 5]
accuracy_scores = []
for p in p_values:
    knn = KNeighborsClassifier(n_neighbors=5, p=p) 
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
print("Accuracy scores for each p value:")
for p, accuracy in zip(p_values, accuracy_scores):
    print(f"p={p}: {accuracy}")
print()




# Applying KNN to a regression problem
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k_values = [1, 3, 5, 7, 9]
mse_scores = []
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
plt.scatter(k_values, mse_scores)
plt.xlabel("k")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs. k Values")
plt.show()
k_values = range(1, 30)
mse_scores = []
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
best_k = k_values[mse_scores.index(min(mse_scores))]
print("The best value of k is:", best_k)




#-------------------Lab Activiy 1-------------------
# KNN for Credit Card Fraud Detection
import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Dataset downloaded to:", path)
df = pd.read_csv(os.path.join(path, "creditcard.csv"))
print(df["Class"].value_counts())
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
print("After SMOTE:", pd.Series(y_train_sm).value_counts().to_dict())
knn_base = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_base.fit(X_train_scaled, y_train)
y_pred_base = knn_base.predict(X_test_scaled)
knn_sm = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_sm.fit(X_train_sm, y_train_sm)
y_pred_sm = knn_sm.predict(X_test_scaled)
fraud_idx = y_train[y_train == 1].index
normal_idx = y_train[y_train == 0].sample(30000, random_state=42).index
sample_idx = fraud_idx.union(normal_idx)
X_sample = scaler.transform(X_train.loc[sample_idx])
y_sample = y_train.loc[sample_idx]
pipe = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier())
])
param_grid = {"knn__n_neighbors": [1, 3, 5, 7, 9, 11]}
grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1", n_jobs=1)
grid.fit(X_sample, y_sample)
best_k = grid.best_params_["knn__n_neighbors"]
print("The best value of k is:", best_k)
knn_best = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn_best.fit(X_train_sm, y_train_sm)
y_pred_best = knn_best.predict(X_test_scaled)
results = pd.DataFrame({
    "Model": ["KNN k=5 (no SMOTE)", "KNN k=5 (SMOTE)", f"KNN k={best_k} (SMOTE + tuned)"],
    "Accuracy": [accuracy_score(y_test, y_pred_base),
                 accuracy_score(y_test, y_pred_sm),
                 accuracy_score(y_test, y_pred_best)],
    "F1-Score": [f1_score(y_test, y_pred_base),
                 f1_score(y_test, y_pred_sm),
                 f1_score(y_test, y_pred_best)],
})
print(results.round(4))
print("\nDetailed report (tuned model):")
print(classification_report(y_test, y_pred_best))









