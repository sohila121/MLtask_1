import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer

df = sns.load_dataset('titanic')

col_name = 'sex'
col_index = df.columns.get_loc(col_name)

encoded_cols = pd.get_dummies(df[col_name], prefix=col_name)

df.drop(columns=[col_name], inplace=True)

for i, new_col in enumerate(encoded_cols.columns):
    df.insert(col_index + i, new_col, encoded_cols[new_col])

print("Encoded Data (One-Hot Encoding)")
print(df.head())

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_analysis = pd.DataFrame({
    'Missing Count': missing_count,
    'Percentage %': missing_percent
})

print("Missing Values Analysis")
print(missing_analysis[missing_analysis['Missing Count'] > 0])

imputer = SimpleImputer(strategy='median')
df['age'] = imputer.fit_transform(df[['age']]).ravel()

print("Imputation Done for Age")
print(f"Missing values in Age: {df['age'].isnull().sum()}")