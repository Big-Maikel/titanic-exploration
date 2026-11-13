
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("data/train.csv")

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

categorical_features = ["Sex", "Embarked"]

numeric_features = [
    "Pclass", "Age", "SibSp", "Parch", "Fare"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numeric", "passthrough", numeric_features)
    ]
)

mlflow.set_experiment("Titanic Decision Tree")

for depth in range(1, 11):

    with mlflow.start_run(run_name=f"Decision Tree depth {depth}"):

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                criterion="entropy",
                max_depth=depth,
                random_state=42
            ))
        ])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", "Decision Tree")
        mlflow.log_param("criterion", "entropy")
        mlflow.log_param("max_depth", depth)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            name="decision_tree_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"\nTree depth: {depth}")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1:.3f}")

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred))


print("\nAll experiments are completed!")

print("\nCreating visualization for tree depth 3...")

best_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,
        random_state=42
    ))
])

best_model.fit(X_train, y_train)

tree_model = best_model.named_steps["classifier"]

feature_names = (
    best_model.named_steps["preprocessor"].get_feature_names_out()
)

plt.figure(figsize=(22, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Titanic Decision Tree - Depth 3")

plt.tight_layout()

plt.savefig("decision_tree_depth_3.png", dpi=300, bbox_inches="tight")

plt.show()

print("\nVisualization saved as decision_tree_depth_3.png")
