

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. Load Dataset
iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("\n===== FIRST 5 ROWS =====")
print(df.head())

# 3. Basic Information
print("\n===== DATA INFO =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

# 4. Check Missing Values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# 5. OUTLIER DETECTION (QUANTILE METHOD - IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Define limits
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = ((df < lower_bound) | (df > upper_bound))
print("\n===== OUTLIERS COUNT =====")
print(outliers.sum())

# 6. REMOVE OUTLIERS
df_clean = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]
print("\nShape before removing outliers:", df.shape)
print("Shape after removing outliers:", df_clean.shape)

# 7. DATA VISUALIZATION
# Histogram
df_clean.hist(figsize=(10,8))
plt.suptitle("Feature Distribution")
plt.show()

# Scatter Plot
plt.scatter(df_clean['sepal length (cm)'], df_clean['petal length (cm)'])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Scatter Plot")
plt.show()

# 8. CORRELATION MATRIX
print("\n===== CORRELATION =====")
print(df_clean.corr())

# 9. FEATURE & TARGET SPLIT
X = df_clean.drop('target', axis=1)
y = df_clean['target']

# 10. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 11. FEATURE SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 12. MODEL TRAINING
model = LogisticRegression()
model.fit(X_train, y_train)

# 13. PREDICTION
y_pred = model.predict(X_test)

# 14. EVALUATION
accuracy = accuracy_score(y_test, y_pred)
print("\n===== ACCURACY =====")
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n===== CONFUSION MATRIX =====")
print(cm)

# 15. FINAL RESULT
print("\n===== WORKFLOW COMPLETED SUCCESSFULLY =====")

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

plt.hist(df['sepal length (cm)'])
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()

import seaborn as sns

sns.boxplot(x=df['sepal length (cm)'])
plt.title("Boxplot of Sepal Length")
plt.show()

df['target'] = iris.target

sns.pairplot(df, hue='target')
plt.show()
