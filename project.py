import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

num_samples = 5000

machine_types = np.random.choice(['L', 'M', 'H'], size=num_samples)

air_temp = np.random.normal(300, 2, num_samples)

process_temp = air_temp + np.random.normal(10, 1, num_samples)

rotational_speed = np.random.normal(1500, 300, num_samples)

torque = np.random.normal(40, 10, num_samples)

tool_wear = np.random.randint(0, 250, num_samples)

failure = (
    (tool_wear > 200) |
    (torque > 60) |
    (rotational_speed < 1000)
).astype(int)

df = pd.DataFrame({
    'Type': machine_types,
    'Air temperature': air_temp,
    'Process temperature': process_temp,
    'Rotational speed': rotational_speed,
    'Torque': torque,
    'Tool wear': tool_wear,
    'Machine failure': failure
})

print("\n================ DATASET HEAD ================\n")
print(df.head())

print("\n================ DATASET INFO ================\n")
print(df.info())

print("\n================ MISSING VALUES ================\n")
print(df.isnull().sum())

le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

print("\n================ STATISTICAL SUMMARY ================\n")
print(df.describe())

plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title('Machine Failure Distribution')
plt.show()

plt.figure(figsize=(10,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

dt_model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

def evaluate_model(model_name, y_true, y_pred):

    print("\n================================================")
    print(f"{model_name} RESULTS")
    print("================================================")

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.show()

evaluate_model(
    'Logistic Regression',
    y_test,
    y_pred_lr
)

evaluate_model(
    'Decision Tree',
    y_test,
    y_pred_dt
)

evaluate_model(
    'Random Forest',
    y_test,
    y_pred_rf
)

lr_acc = accuracy_score(y_test, y_pred_lr)
dt_acc = accuracy_score(y_test, y_pred_dt)
rf_acc = accuracy_score(y_test, y_pred_rf)

accuracy_df = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest'
    ],
    'Accuracy': [
        lr_acc,
        dt_acc,
        rf_acc
    ]
})

print("\n================ MODEL COMPARISON ================\n")
print(accuracy_df)

plt.figure(figsize=(8,5))

sns.barplot(
    x='Model',
    y='Accuracy',
    data=accuracy_df
)

plt.title('Model Accuracy Comparison')
plt.ylim(0.8, 1.0)

plt.show()

importances = rf_model.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n================ FEATURE IMPORTANCE ================\n")
print(feature_df)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_df
)

plt.title('Feature Importance - Random Forest')

plt.show()

best_model = accuracy_df.sort_values(
    by='Accuracy',
    ascending=False
).iloc[0]

print("\n================================================")
print("BEST MODEL")
print("================================================")

print(f"\nModel Name : {best_model['Model']}")
print(f"Accuracy   : {best_model['Accuracy']:.4f}")

sample_data = np.array([
    [1, 300, 310, 1200, 65, 220]
])

sample_scaled = scaler.transform(sample_data)

prediction = rf_model.predict(sample_scaled)

print("\n================ SAMPLE PREDICTION ================\n")

if prediction[0] == 1:
    print("Machine Failure Predicted")
else:
    print("Machine Working Normally")

import joblib

joblib.dump(rf_model, 'predictive_maintenance_model.pkl')

print("\nModel saved successfully!")