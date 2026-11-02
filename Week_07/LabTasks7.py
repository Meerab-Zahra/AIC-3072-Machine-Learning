# ------------Activity 1------------
# ------------Task 1------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
water = pd.read_csv('water_potability.csv')
print("Shape:", water.shape)
print(water.head(10))
print("\nMissing values:\n", water.isnull().sum())
print("\nClass distribution:\n", water['Potability'].value_counts())
print(water['Potability'].value_counts(normalize=True).round(3))
sns.countplot(data=water, x='Potability')
plt.show()
 
# ------------Task 2------------
X = water.drop('Potability', axis=1)
y = water['Potability']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape)
 
# ------------Task 3------------
train_median = X_train.median()
X_train = X_train.fillna(train_median)
X_test = X_test.fillna(train_median)
print("Missing in train:", X_train.isnull().sum().sum())
print("Missing in test :", X_test.isnull().sum().sum())
 
# ------------Task 4------------
lr  = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
svc = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
rf  = RandomForestClassifier(n_estimators=100, random_state=42)
estimators = [('lr', lr), ('svc', svc), ('rf', rf)]
lr.fit(X_train, y_train)
svc.fit(X_train, y_train)
rf.fit(X_train, y_train)
vote_hard = VotingClassifier(estimators, voting='hard').fit(X_train, y_train)
vote_soft = VotingClassifier(estimators, voting='soft').fit(X_train, y_train)
 
# ------------Task 5------------
base_tree = DecisionTreeClassifier(random_state=42)
bag_10 = BaggingClassifier(base_tree, n_estimators=10, max_samples=0.5,
                           bootstrap=True, random_state=42).fit(X_train, y_train)
bag_100 = BaggingClassifier(base_tree, n_estimators=100, max_samples=0.5,
                            bootstrap=True, random_state=42).fit(X_train, y_train)
paste_100 = BaggingClassifier(base_tree, n_estimators=100, max_samples=0.5,
                              bootstrap=False, random_state=42).fit(X_train, y_train)

single_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
 
# ------------Task 6------------
all_models = {
    'Logistic Regression': lr,
    'SVC': svc,
    'Random Forest': rf,
    'Single Decision Tree': single_tree,
    'Voting (hard)': vote_hard,
    'Voting (soft)': vote_soft,
    'Bagging (10 trees)': bag_10,
    'Bagging (100 trees)': bag_100,
    'Pasting (100 trees)': paste_100,
}
rows = []
for name, model in all_models.items():
    pred = model.predict(X_test)
    rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1-score': f1_score(y_test, pred, zero_division=0),
    })
results_1 = pd.DataFrame(rows).set_index('Model').round(4)
results_1 = results_1.sort_values('Accuracy', ascending=False)
print(results_1)
best_name = results_1.index[0]
print("Best model by accuracy:", best_name)
ConfusionMatrixDisplay.from_estimator(
    all_models[best_name], X_test, y_test, display_labels=['Not safe (0)', 'Safe (1)'], cmap='Blues'
)
plt.title(f'Confusion matrix: {best_name}')
plt.show()
singles = ['Logistic Regression', 'SVC', 'Random Forest', 'Single Decision Tree']
ensembles = [m for m in results_1.index if m not in singles]
best_single = results_1.loc[singles, 'Accuracy'].idxmax()
best_ens = results_1.loc[ensembles, 'Accuracy'].idxmax()
print(f"Best single model : {best_single} ({results_1.loc[best_single, 'Accuracy']:.4f})")
print(f"Best ensemble     : {best_ens} ({results_1.loc[best_ens, 'Accuracy']:.4f})")
print("Ensemble beat the best single model?",
      results_1.loc[best_ens, 'Accuracy'] > results_1.loc[best_single, 'Accuracy'])
 



# ------------Activity 2------------
# ------------Task 1------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,                           f1_score, roc_auc_score)
from xgboost import XGBClassifier
telco = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Shape:", telco.shape)
print(telco.head(10))
print("\nMissing values (NaN):\n", telco.isnull().sum())
print("\nChurn distribution:\n", telco['Churn'].value_counts())
print(telco['Churn'].value_counts(normalize=True).round(3))
sns.countplot(data=telco, x='Churn')
plt.show()

# ------------Task 2------------
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
print("Blank TotalCharges:", telco['TotalCharges'].isnull().sum())
print(telco.loc[telco['TotalCharges'].isnull(), ['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges']])
telco['TotalCharges'] = telco['TotalCharges'].fillna(0)
print("Missing after fix:", telco['TotalCharges'].isnull().sum())
 
# ------------Task 3------------
telco = telco.drop('customerID', axis=1)
telco['Churn'] = telco['Churn'].map({'Yes': 1, 'No': 0})
X = telco.drop('Churn', axis=1)
y = telco['Churn']
cat_cols = X.select_dtypes(include='object').columns.tolist()
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(X_train.shape, X_test.shape)
print("Churn rate train:", round(y_train.mean(), 3), "| test:", round(y_test.mean(), 3))
 
# ------------Task 4------------
gb  = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
ada = AdaBoostClassifier(random_state=42).fit(X_train, y_train)
xgb_default = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_train, y_train)
 
def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        'Accuracy': accuracy_score(y, pred),
        'Precision': precision_score(y, pred),
        'Recall': recall_score(y, pred),
        'F1-score': f1_score(y, pred),
        'ROC-AUC': roc_auc_score(y, proba),
    }
 
default_results = pd.DataFrame({
    'Gradient Boosting': evaluate(gb, X_test, y_test),
    'AdaBoost': evaluate(ada, X_test, y_test),
    'XGBoost (default)': evaluate(xgb_default, X_test, y_test),
}).T.round(4)
default_results
 
# ------------Task 5------------
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
}
grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best CV F1-score:", round(grid.best_score_, 4))
 
# ------------Task 6------------
xgb_tuned = grid.best_estimator_
compare = pd.DataFrame({
    'XGBoost (default)': evaluate(xgb_default, X_test, y_test),
    'XGBoost (tuned)': evaluate(xgb_tuned, X_test, y_test),
}).T.round(4)
print(compare)
importances = pd.Series(xgb_tuned.feature_importances_, index=X_train.columns)
top5 = importances.sort_values(ascending=False).head(5)
plt.figure(figsize=(8, 4))
top5.sort_values().plot(kind='barh', color='teal')
plt.title('Top 5 feature importances (tuned XGBoost)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()
baseline_acc = 1 - y_test.mean()
print(f"Always predicting 'No churn' gives accuracy = {baseline_acc:.3f}, but recall = 0")



 
# ------------Activity 3------------
# ------------Task 1------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ins = pd.read_csv('insurance.csv')
print("Shape:", ins.shape)
print(ins.head(10))
print("\nMissing values:\n", ins.isnull().sum())
print("\nSummary statistics of charges:\n", ins['charges'].describe())
 
# ------------Task 2------------
ins['sex'] = ins['sex'].map({'male': 0, 'female': 1})
ins['smoker'] = ins['smoker'].map({'no': 0, 'yes': 1})
ins = pd.get_dummies(ins, columns=['region'], drop_first=True)
ins = ins.astype({c: float for c in ins.columns if ins[c].dtype == bool})
X = ins.drop('charges', axis=1)
y = ins['charges']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(X_train.shape, X_test.shape)
 
# ------------Task 3------------
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
gb_reg = GradientBoostingRegressor(random_state=42)
ridge.fit(X_train, y_train)
rf_reg.fit(X_train, y_train)
gb_reg.fit(X_train, y_train)
 
# ------------Task 4------------
stack_reg = StackingRegressor(
    estimators=[('ridge', ridge), ('rf', rf_reg), ('gb', gb_reg)],
    final_estimator=LinearRegression(),
    cv=5
)
stack_reg.fit(X_train, y_train)
 
# ------------Task 5------------
def reg_scores(model):
    pred = model.predict(X_test)
    return {
        'MAE': mean_absolute_error(y_test, pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
        'R2': r2_score(y_test, pred),
    }
results_3 = pd.DataFrame({
    'Ridge': reg_scores(ridge),
    'Random Forest': reg_scores(rf_reg),
    'Gradient Boosting': reg_scores(gb_reg),
    'Stacking': reg_scores(stack_reg),
}).T.round(4)
results_3
 
# ------------Task 6------------
coefs = pd.Series(stack_reg.final_estimator_.coef_, index=['Ridge', 'Random Forest', 'Gradient Boosting'])
print("Final estimator coefficients:\n", coefs.round(4))
print("Intercept:", round(stack_reg.final_estimator_.intercept_, 2))
print("\nBase model trusted most:", coefs.idxmax())
y_pred_stack = stack_reg.predict(X_test)
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_stack, alpha=0.6, color='teal')
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, 'r--', label='Perfect prediction')
plt.xlabel('Actual charges')
plt.ylabel('Predicted charges')
plt.title('Stacking: actual vs predicted charges')
plt.legend()
plt.show()
 