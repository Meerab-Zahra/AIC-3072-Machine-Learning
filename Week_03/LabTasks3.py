# Activity 1: Customer Purchase Prediction Using Logistic Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay)
data = pd.read_csv("DataSets/Social_Network_Ads.csv")
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data.drop('User ID', axis=1, inplace=True)
X = data[['Gender', 'Age', 'EstimatedSalary']].values
y = data['Purchased'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay( confusion_matrix=cm, display_labels=['Not Purchased', 'Purchased'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='red', label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Purchase Prediction')
plt.legend()
plt.show()



# Activity 2: Admission Prediction using Linear and Polynomial Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
data = pd.read_csv("DataSets/Admission_Predict.csv")
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']].values
y = data['Chance of Admit '].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
rmse_lin = np.sqrt(metrics.mean_squared_error(y_test, y_pred_lin))
r2_lin = metrics.r2_score(y_test, y_pred_lin)
print("---- Linear Regression ----")
print("RMSE:", rmse_lin)
print("R2 Score:", r2_lin)
results = {}
for degree in [2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    rmse_poly = np.sqrt(metrics.mean_squared_error(y_test, y_pred_poly))
    r2_poly = metrics.r2_score(y_test, y_pred_poly)
    results[degree] = (rmse_poly, r2_poly)
    print(f"---- Polynomial Regression (degree={degree}) ----")
    print("RMSE:", rmse_poly)
    print("R2 Score:", r2_poly)
feature_index = 5
sort_idx = np.argsort(X_test[:, feature_index])
# Plot actual data and polynomial regression curves
plt.figure(figsize=(8, 5))
plt.scatter(X_test[:, feature_index], y_test, color='blue', label='Actual Data', alpha=0.5)
# Plot polynomial regression curves for different degrees
for degree in [2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    plt.plot(X_test[sort_idx, feature_index], y_pred_poly[sort_idx], label=f'Degree {degree}')
plt.xlabel('CGPA')
plt.ylabel('Chance of Admit')
plt.title('Polynomial Regression Fits (degree=2,3,4)')
plt.legend()
plt.show()
print("\nSummary (RMSE, R2):")
print("Linear:", rmse_lin, r2_lin)
for degree, (r, s) in results.items():
    print(f"Degree {degree}:", r, s)