import pandas as pd

df = pd.read_csv("data/train.csv")

print(df.head())
print(df.shape)
print(df.info())

print(df.isnull().sum())

print("\n--- Survival ---")

print(df["Survived"].value_counts())
print(df["Survived"].value_counts(normalize=True) * 100)

print("\n--- Survival per sex ---")

survival_by_sex = df.groupby("Sex")["Survived"].mean() * 100

print(survival_by_sex)

print("\n--- Survival per passenger class ---")

survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100

print(survival_by_class.round(2))

print("\n--- Survival per sex and passenger class ---")

survival_by_sex_class = df.pivot_table(
    values="Survived",
    index="Pclass",
    columns="Sex",
    aggfunc="mean"
) * 100

print(survival_by_sex_class.round(2))