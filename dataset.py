import pandas as pd

df = pd.read_csv("Data/risk_factors_cervical_cancer.csv")
df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1, inplace=True)
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv("Data/cleaned_dataset.csv", index=False)
