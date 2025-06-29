import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Load Titanic dataset
df = sns.load_dataset('titanic')

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

# Visualize missing values
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Fix: Avoid chained assignment
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df.drop(columns=['deck'], inplace=True)

# Univariate Analysis - Numerical
df.hist(figsize=(12,10), bins=30, color='skyblue', edgecolor='black')
plt.suptitle('Distributions of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()

# Boxplot - Outliers in Age
plt.figure(figsize=(8,6))
sns.boxplot(x=df['age'], color='orange')
plt.title('Boxplot of Age')
plt.show()

# Univariate Analysis - Categorical
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sex', hue='sex', palette='pastel', legend=False)
plt.title('Distribution of Sex')
plt.show()

# Categorical vs Target
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sex', hue='survived', palette='Set2')
plt.title('Survival by Sex')
plt.show()

# Numerical vs Target
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='survived', y='age', hue='survived', palette='coolwarm', legend=False)
plt.title('Age Distribution by Survival')
plt.show()

# Encode categorical for correlation
df_encoded = df.copy()
df_encoded['sex'] = df_encoded['sex'].map({'male': 0, 'female': 1})
df_encoded['embarked'] = df_encoded['embarked'].map({'S':0, 'C':1, 'Q':2})
df_encoded['class'] = df_encoded['class'].map({'Third':3, 'Second':2, 'First':1})
df_encoded['who'] = df_encoded['who'].map({'man':0, 'woman':1, 'child':2})
df_encoded['adult_male'] = df_encoded['adult_male'].astype(int)
df_encoded['alone'] = df_encoded['alone'].astype(int)

# Drop non-numeric columns before correlation
df_corr = df_encoded.select_dtypes(include=[np.number])

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Outlier detection in Fare
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['fare'] < Q1 - 1.5 * IQR) | (df['fare'] > Q3 + 1.5 * IQR)]
print("\nNumber of Fare Outliers:", len(outliers))

# Pairplot - Multivariate relationship
sns.pairplot(df.dropna(), hue='survived', vars=['age', 'fare', 'pclass'], palette='husl')
plt.suptitle("Pairwise Plots", y=1.02)
plt.show()

# Crosstab Example
print("\nSurvival rate by Pclass:")
print(pd.crosstab(df['pclass'], df['survived'], normalize='index'))
