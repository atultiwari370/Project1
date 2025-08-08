"""
Titanic Survival Prediction - Simple end-to-end script

Usage:
    python titanic_model.py

This script:
 - loads dataset/dataset.csv (sample included)
 - does basic cleaning and feature engineering
 - trains LogisticRegression
 - prints accuracy and example predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path("dataset/titanic_sample.csv")  # Change to titanic.csv if you put full dataset

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print("Data loaded:", df.shape)
    return df

def preprocess(df):
    df = df.copy()
    # Keep useful columns
    cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    # Fill missing Age with median
    df["Age"] = df["Age"].fillna(df["Age"].median())
    # Fill Embarked with most common
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # Fill Fare
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    # Encode Sex and Embarked
    le_sex = LabelEncoder()
    df["Sex"] = le_sex.fit_transform(df["Sex"].astype(str))
    le_emb = LabelEncoder()
    df["Embarked"] = le_emb.fit_transform(df["Embarked"].astype(str))
    return df

def train_model(df):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    return model, X_test, y_test, preds

def example_predictions(model, X_test):
    print("\nSample predictions (first 10):")
    sample = X_test.head(10).copy()
    sample["predicted_survived"] = model.predict(sample)
    print(sample)

def main():
    df = load_data()
    df = preprocess(df)
    model, X_test, y_test, preds = train_model(df)
    example_predictions(model, X_test)

if __name__ == "__main__":
    main()
