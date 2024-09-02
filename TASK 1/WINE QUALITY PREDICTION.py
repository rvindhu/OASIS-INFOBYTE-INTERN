import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from warnings import filterwarnings

# Ignore warnings
filterwarnings(action='ignore')


# Path to the specific file
file_path = r"C:\Users\rvind\Downloads\WineQT.csv"
print(file_path)

# Load the dataset
wine = pd.read_csv(file_path)

# Display dataset information
print(wine.shape)
print(wine.info())
print(wine.describe(include='all'))
print(wine.isna().sum())
print(wine.corr())
print(wine.groupby('quality').mean())

# Plot histograms
wine.hist(figsize=(12, 10))
plt.suptitle('Histogram of Each Numeric Column')
plt.show()

plt.figure(figsize=(12, 10))
sns.pairplot(wine, diag_kind='kde')
plt.suptitle('Pairwise Relationships in Dataset')
plt.show()

# Scatter plots for each feature vs. quality
for col in wine.columns:
    if col != 'quality':
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=col, y='quality', data=wine)
        plt.title(f'Relationship between {col} and Quality')
        plt.show()

# Bar plots for each feature vs. quality
features = [col for col in wine.columns if col != 'quality']
num_rows = 4
num_cols = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12))

for i, feature in enumerate(features):
    row = i // num_cols
    col = i % num_cols
    sns.barplot(x='quality', y=feature, data=wine, ax=axes[row, col])
    axes[row, col].set_title(f'Quality vs {feature}')
    axes[row, col].set_xlabel('Quality')
    axes[row, col].set_ylabel(feature)

# Hide any unused subplots
for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axes[j // num_cols, j % num_cols])

plt.tight_layout()
plt.show()

# Preprocessing: Binning and encoding the target variable
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
print(wine['quality'].value_counts())

# Prepare features and target variable
X = wine.drop('quality', axis=1)
y = wine['quality']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize different regression models
models = {
    'Random Forest': RandomForestRegressor(),
    'Linear Regression': LinearRegression(),
    'Support Vector Machine': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor()
}

# Train and evaluate each model
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse:.2f}")
    print(f"{name} - R-squared: {r2:.2f}")
