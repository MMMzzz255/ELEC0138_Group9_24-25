import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)


def load_data(path):
    """
    Load and clean the dataset.
    :param path: Path to the dataset
    :return: Cleaned DataFrame
    """
    df = pd.read_csv(path)
    df.replace([-np.inf, np.inf], np.nan, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, df.nunique() > 1]
    return df


def preprocess_features(df):
    """
    Preprocess features by imputing missing values.
    :param df: DataFrame containing features
    :return: Tuple of imputed features DataFrame, target Series, and imputer object
    """
    X = df.drop(columns=["Target"])
    y = df["Target"]
    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_imputed.dropna(inplace=True)
    y = y.loc[X_imputed.index]
    return X_imputed, y, imputer


def scale_features(X):
    """
    Scale features using StandardScaler.
    :param X: Features to scale
    :return: Scaled features and the scaler object
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled, scaler


def train_svm_with_grid_search(X_train, y_train):
    """
    Train SVM model with grid search for hyperparameter tuning.
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained SVM model and grid search object
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search


def get_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for the model.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Dictionary of evaluation metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }


def evaluate_model(model, X_test, y_test, cm_path="../result/confusion_matrix_svm.png",
                   metrics_path="../result/metrics_svm.csv"):
    """
    Evaluate the model using confusion matrix and classification report.
    :param model: Trained SVM model
    :param X_test: Test features
    :param y_test: Test labels
    :param cm_path: Path to save the confusion matrix plot
    :param metrics_path: Path to save the metrics CSV
    :return: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    metrics = get_metrics(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Phishing"], yticklabels=["Benign", "Phishing"])
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path)
    plt.close()

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

    return metrics


def save_artifacts(model, scaler, imputer, top_features, path="../model"):
    """
    Save the trained model, scaler, imputer, and top features to disk.
    :param model: Trained SVM model
    :param scaler: StandardScaler object
    :param imputer: SimpleImputer object
    :param top_features: List of top feature names
    :param path: Directory to save the artifacts
    """
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(path, "svm_scaler.pkl"))
    joblib.dump(imputer, os.path.join(path, "svm_imputer.pkl"))
    joblib.dump(top_features, os.path.join(path, "svm_top_features.pkl"))


def main():
    # Load the preprocessed data
    df = load_data("../data/processed_phishing_data.csv")

    # Preprocess features
    X_imputed, y, imputer = preprocess_features(df)

    # Scale features for SVM
    X_scaled, scaler = scale_features(X_imputed)

    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train SVM
    svm_model, _ = train_svm_with_grid_search(X_train, y_train)

    # Evaluate and save
    metrics = evaluate_model(svm_model, X_test, y_test)

    # Save model artifacts
    save_artifacts(svm_model, scaler, imputer, X_imputed.columns.tolist())

    # Output summary
    print("\nSVM Model Metrics Summary:")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    main()
