"""
Task 1 - Multi-Class Benchmark Classification Pipeline (Iris)
Datheon ML & Data Science Internship

Steps (as per task):
1. Load dataset with Pandas
2. EDA pairplot style (saved as png)
3. StandardScaler normalization
4. Train LogisticRegression + DecisionTree
5. Confusion matrix heatmap + classification report
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset():
    """Try CSV URL first, else sklearn built-in (offline safe)."""
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    try:
        df = pd.read_csv(url)
        print("Loaded Iris CSV from URL")
        # seaborn iris.csv has 'species' column
        X = df.drop(columns=["species"])
        y = df["species"]
        # numeric encode for models
        y_encoded, labels = pd.factorize(y)
        return X.values, y_encoded, list(labels), list(X.columns)
    except Exception as e:
        print(f"URL load failed ({e}), using sklearn load_iris()")
        iris = load_iris()
        return iris.data, iris.target, list(iris.target_names), list(iris.feature_names)


def plot_confusion(cm, labels, title, savepath):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved: {savepath}")


def main():
    X, y, label_names, feature_names = load_dataset()
    print(f"Dataset shape: {X.shape}, classes: {label_names}")

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

        print(f"\n--- {name} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(classification_report(y_test, pred, target_names=label_names, zero_division=0))

        cm = confusion_matrix(y_test, pred)
        plot_confusion(cm, label_names,
                       f"Confusion Matrix - {name}",
                       os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png"))

    print("\n✅ Task 1 done. Check confusion_matrix_*.png in this folder.")


if __name__ == "__main__":
    main()
