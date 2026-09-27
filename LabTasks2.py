# Loading dataset & printing it
import pandas as pd
df=pd.read_csv("DataSets/headbrain.csv")
X = df['Head Size(cm^3)'].values
y = df['Brain Weight(grams)'].values
print(df.head())
# Calculating parameters
import numpy as np
X_sum = np.sum(X)
X_squared_sum = np.sum(X * X)
n = len(X)
print("Sum of X:", X_sum)
print("Sum of X squared:", X_squared_sum)
print("Length of X:", n)
# Estimating parameters
numer=n*np.sum(X*y)-np.sum(X)*np.sum(y)
denom=n*np.sum(X*X)-(np.sum(X))**2
w1=numer/denom
w0=(np.sum(y)-w1*(np.sum(X)))/n
# Plotting regresion line
import matplotlib.pyplot as plt
max_x = np.max(X)
min_x = np.min(X)
x1 = np.linspace(min_x, max_x)
y1 = w0 + w1 * x1
plt.plot(x1, y1, color='red', label='Regression Line')
plt.scatter(X, y, c='green', label='Scatter Plot')
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()
# Calculating RMSE
rmse = 0
for i in range(n):
    y_pred = w0 + w1 * X[i]
    rmse += (y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/n)
print("RMSE=",rmse)
# Calculating R2 score
ss_tot = 0
ss_res = 0
y_mean=np.mean(y)
for i in range(n):
    y_pred = w0 + w1 * X[i]
    ss_tot += (y[i] -y_mean) ** 2
    ss_res += (y[i] - y_pred) ** 2
r2 = 1 - (ss_res/ss_tot)
print("R2 Score=",r2)



# Multivariate Linear Regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
df = pd.read_csv("DataSets/Admission_Predict.csv")
df.drop("Serial No.", axis=1, inplace=True)
y = df["Chance of Admit "]
df.drop("Chance of Admit ", axis=1, inplace=True)
print(df.head())
simple_lr = LinearRegression()
simple_lr.fit(df[["GRE Score"]], y)
simple_pred = simple_lr.predict(df[["GRE Score"]])
plt.scatter(df["GRE Score"],y,color="green",label="Actual Data",alpha=0.5)
plt.plot(df["GRE Score"],simple_pred,color="red",linewidth=3,label="Regression Line")
plt.xlabel("GRE Score")
plt.ylabel("Admission Chance")
plt.title("GRE Score vs Admission Chance")
plt.legend()
plt.show()
x_train, x_test, y_train, y_test = train_test_split(df,y,test_size=0.2,random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
print("RMSE:", rmse)
coefficients = lr.coef_
features = df.columns
plt.figure(figsize=(8, 5))
plt.barh(features,coefficients,color="teal")
plt.xlabel("Coefficient Value (Weight)")
plt.title("Feature Importance in Multivariate Linear Regression")
plt.axvline(x=0,color="black",linewidth=0.8)
plt.show()



# Logistic Regression from scratch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv('DataSets/Social_Network_Ads.csv')
labelencoder = LabelEncoder()
data['Gender'] = labelencoder.fit_transform(data['Gender'])  # Male=1, Female=0
X = data[['Gender', 'Age', 'EstimatedSalary']].values  
y = data['Purchased'].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.lr = learning_rate
        self.iterations = num_iterations
        self.weights = None
        self.bias = None
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)    
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))



# Logistic Regression using sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv("DataSets/Social_Network_Ads.csv")
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
X = data[['Gender', 'Age', 'EstimatedSalary']].values
y = data['Purchased'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



# Polynomial Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv("DataSets/Position_Salaries.csv")
X = data[["Level"]].values
y = data["Salary"].values
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred = poly_model.predict(X_poly)
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Polynomial Fit (deg=4)")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.legend()
plt.show()



# Activity 1 -  Predicting a person's medical insurance cost using Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
df = pd.read_csv("DataSets/Medical Cost Personal Datasets.csv")
df.head()
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])          # male=1, female=0
df['smoker'] = le.fit_transform(df['smoker'])    # yes=1, no=0
df['region'] = le.fit_transform(df['region'])    # one code per region
X = df[['age', 'bmi', 'children', 'smoker', 'region']].values
y = df['charges'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R2 Score:", r2)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Fit')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Predicted vs Actual Insurance Costs')
plt.legend()
plt.show()


#Activity 2 - Predicting whether a customer will churn using Logistic Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay)
data = pd.read_csv("DataSets/Telco Customer Churn.csv")
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)
le = LabelEncoder()
data['Contract'] = le.fit_transform(data['Contract'])
data['InternetService'] = le.fit_transform(data['InternetService'])
data['Churn'] = le.fit_transform(data['Churn'])  # Yes=1, No=0
X = data[['tenure', 'MonthlyCharges', 'Contract', 'InternetService']].values
y = data['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='red', label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Customer Churn Prediction')
plt.legend()
plt.show()



# Activity 3 - Car price prediction using Linear and Polynomial Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
data = pd.read_csv("DataSets/Car Price Prediction.csv")
# This dataset has no Kms_Driven, Year, Selling_Price columns so closest available features are horsepower, enginesize, citympg (mileage), curbweight and target is price
X = data[['horsepower', 'enginesize', 'citympg', 'curbweight']].values
y = data['price'].values
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
# Plotting Fitted Curves for Different Degrees vs Horsepower
feature_index = 0  # horsepower column
sort_idx = np.argsort(X_test[:, feature_index])
plt.figure(figsize=(8, 5))
plt.scatter(X_test[:, feature_index], y_test, color='blue', label='Actual Data', alpha=0.5)
for degree in [2, 3, 4]:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    plt.plot(X_test[sort_idx, feature_index], y_pred_poly[sort_idx], label=f'Degree {degree}')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title('Polynomial Regression Fits (degree=2,3,4)')
plt.legend()
plt.show()
print("\nSummary (RMSE, R2):")
print("Linear:", rmse_lin, r2_lin)
for degree, (r, s) in results.items():
    print(f"Degree {degree}:", r, s)

