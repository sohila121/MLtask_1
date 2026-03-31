import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = sns.load_dataset('titanic')

print("DATA STRUCTURE")
df.info()
print(df.head()) 

missing_values = df.isnull().sum()
print("missing values")
print(missing_values[missing_values > 0])

plt.figure(figsize=(8, 4))
sns.boxplot(x=df['fare'], color='skyblue')
plt.title('Outliers in Fare ')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=df, palette='viridis')
plt.title('Data Imbalance ')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x='pclass', y='survived', hue='sex', data=df)
plt.title('Survival Rate by Class and Sex')
plt.show()

plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=['float64', 'int64']) 
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap ')
plt.show()
