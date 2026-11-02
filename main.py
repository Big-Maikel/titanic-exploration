import pandas as pd
import matplotlib.pyplot as plt

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

survival_by_sex_class.plot(kind="bar")

plt.title("Survival by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Sex")
plt.tight_layout()
plt.show()

print("\n--- Amount of passengers per group ---")

group_sizes = df.pivot_table(
    values="Survived",
    index="Pclass",
    columns="Sex",
    aggfunc="count"
)

print(group_sizes)

summary = df.groupby(["Pclass", "Sex"]).agg(
    passengers=("Survived", "size"),
    survivors=("Survived", "sum"),
    survival_rate=("Survived", "mean")
)

summary["survival_rate"] *= 100

print(summary.round(2))

print("\n--- Age statistics ---")
print(df["Age"].describe())

print("\n--- Missing ages ---")
print(df["Age"].isnull().sum())

age_bins = [0, 12, 17, 35, 59, float("inf")]

age_labels = [
    "0-12",
    "13-17",
    "18-35",
    "36-59",
    "60+"
]

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)

print("\n--- Survival per age group ---")

survival_by_age = df.groupby(
    "AgeGroup",
    observed=True
)["Survived"].agg(
    passengers="count",
    survival_rate="mean"
)

survival_by_age["survival_rate"] *= 100

print(survival_by_age.round(2))

print("\n--- Survival per age group and sex ---")

survival_by_age_sex = df.pivot_table(
    values="Survived",
    index="AgeGroup",
    columns="Sex",
    aggfunc="mean"
) * 100

print(survival_by_age_sex.round(2))

print("\n--- Age group and sex: counts and survival ---")

age_sex_summary = df.groupby(
    ["AgeGroup", "Sex"],
    observed=True
).agg(
    passengers=("Survived", "size"),
    survivors=("Survived", "sum"),
    survival_rate=("Survived", "mean")
)

age_sex_summary["survival_rate"] *= 100

print(age_sex_summary.round(2))

survival_by_age_sex.plot(kind="bar")

plt.title("Survival by Age Group and Sex")
plt.xlabel("Age Group")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Sex")
plt.tight_layout()
plt.show()

print("\n--- Passengers per age group and class ---")

age_class_counts = df.pivot_table(
    values="Survived",
    index="AgeGroup",
    columns="Pclass",
    aggfunc="count",
    observed=True
)

print(age_class_counts)

print("\n--- Survival per age group and class ---")

survival_by_age_class = df.pivot_table(
    values="Survived",
    index="AgeGroup",
    columns="Pclass",
    aggfunc="mean",
    observed=True
) * 100

print(survival_by_age_class.round(2))

print("\n--- Survival per age group sex and class ---")

survival_by_age_sex_class = df.pivot_table(
    values="Survived",
    index=["AgeGroup", "Sex"],
    columns="Pclass",
    aggfunc="mean",
    observed=True
) * 100

print(survival_by_age_sex_class.round(2))

print("\n--- Passenger counts per age group, sex and class ---")

passenger_counts = df.pivot_table(
    values="Survived",
    index=["AgeGroup", "Sex"],
    columns="Pclass",
    aggfunc="count",
    observed=True
)

print(passenger_counts)

print("\n--- Embarked distribution ---")
print(df["Embarked"].value_counts())

print("\n--- embarked distribution per class ---")
print(df.groupby("Pclass")["Embarked"].value_counts())

print("\n--- Survival per embarked ---")
survival_by_embarked = df.groupby("Embarked")["Survived"].mean()
print(survival_by_embarked)

print("\n--- Survival per embarked and class ---")
survival_by_embarked_class = df.pivot_table(
    values="Survived",
    index="Embarked",
    columns="Pclass",
    aggfunc="mean"
) * 100

print(survival_by_embarked_class)
