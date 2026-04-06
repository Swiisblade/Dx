# Import Libraries
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print("Original Dataset")
print(df.head())

# 2. Data Preprocessing

# Remove missing values
df = df.dropna()

# Remove duplicates
df = df.drop_duplicates()

print("\nPreprocessed Dataset")
print(df.head())

# 4. Outlier Detection (IQR Method)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

df_clean = df[~((df < (Q1 - 1.5 * IQR)) |
                (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nDataset after Removing Outliers")
print(df_clean.head())

#5. Data Visualization

# Matplotlib Histogram
df_clean.hist()
plt.suptitle("Histogram of Features")
plt.show()

# Seaborn Boxplot
sns.boxplot(data=df_clean)
plt.title("Boxplot for Outliers")
plt.show()
