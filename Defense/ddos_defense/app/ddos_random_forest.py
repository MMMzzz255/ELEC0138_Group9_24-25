import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score


def load_data(path):
    """
    Load and clean the dataset.
    :param path: Path to the dataset
    :return: df: Cleaned DataFrame
    """
    df = pd.read_csv(path)
    df.replace([-np.inf, np.inf], np.nan, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)  # Drop columns that are entirely NaN
    df = df.loc[:, df.nunique() > 1]  # Drop columns with only one unique value
    return df


def preprocess_features(df):
    """
    Preprocess features by imputing missing values on the full set of columns.
    (Used only to get a 'cleaned' DataFrame for top-feature selection.)
    :param df: DataFrame containing features
    :return: X_imputed: Imputed features DataFrame, y: Target Series
    """
    X = df.drop(columns=["Class", "Label"], errors='ignore')
    y = df["Class"] if "Class" in df.columns else df.iloc[:, -1]

    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Ensure rows align
    X_imputed.dropna(inplace=True)
    y = y.loc[X_imputed.index]

    return X_imputed, y  # Not returning imputer; we'll re-fit on top 8 features.


def select_top_features(X, y, top_n=8):
    """
    Select top N features using Random Forest feature importance.
    :param X: Features DataFrame
    :param y: Target Series
    :param top_n: Number of top features to select
    :return: X_top: DataFrame with top features, top_features: List of top feature names
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()

    print("Top Features:", top_features)
    X_top = X[top_features]
    return X_top, top_features


def train_rf_with_grid_search(X_train, y_train):
    """
    Train Random Forest model with grid search for hyperparameter tuning.
    :param X_train: Training features
    :param y_train: Training target
    :return: Trained Random Forest model
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_


def get_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for the model.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Dictionary of metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }


def evaluate_model(model, X_test, y_test,
                   cm_path="../result/confusion_matrix_rf.png",
                   metrics_path="../result/metrics_rf.csv"):
    """
    Evaluate the model using confusion matrix and classification report.
    :param model: Trained model
    :param X_test: Test features
    :param y_test: Test labels
    :param cm_path: Path to save confusion matrix
    :param metrics_path: Path to save metrics
    :return: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    metrics = get_metrics(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path)
    plt.close()

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    return metrics


def save_artifacts(model, scaler, imputer, top_features, path="../model"):
    """
    Save the trained model, scaler, imputer, and top feature names to disk.
    :param model: Trained model
    :param scaler: Scaler object
    :param imputer: Imputer object
    :param top_features: List of top feature names
    :param path: Directory to save artifacts
    """
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "rf_model.pkl"))
    joblib.dump(scaler, os.path.join(path, "rf_scaler.pkl"))
    joblib.dump(imputer, os.path.join(path, "rf_imputer.pkl"))
    joblib.dump(top_features, os.path.join(path, "rf_top_features.pkl"))


def save_cleaned_top_features(X_top_scaled, y, output_path="../data/cleaned_top_8_features.csv"):
    """
    Save the final top features (already scaled) and target variable to a CSV file.
    :param X_top_scaled: Scaled features DataFrame
    :param y: Target Series
    :param output_path: Path to save the cleaned top features CSV file
    """
    df_cleaned = X_top_scaled.copy()
    df_cleaned['Target'] = y.values
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)


def main():
    """
    Main function:
    """
    csv_path = "../data/preprocessed_ddos_data.csv"
    df = load_data(csv_path)

    if 'Class' in df.columns:
        df['Class'] = df['Class'].apply(lambda x: 1 if x == 'Attack' else 0)

    X_imputed_full, y_full = preprocess_features(df)

    X_top_candidate, top_features = select_top_features(X_imputed_full, y_full, top_n=8)

    X_final = X_top_candidate.copy()
    y_final = y_full.loc[X_final.index]

    imputer = SimpleImputer(strategy="mean")

    X_imputed_8 = pd.DataFrame(imputer.fit_transform(X_final), columns=X_final.columns)

    scaler = StandardScaler()
    X_scaled_8 = pd.DataFrame(scaler.fit_transform(X_imputed_8), columns=X_imputed_8.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled_8, y_final, test_size=0.2, random_state=42)

    model = train_rf_with_grid_search(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    save_artifacts(model, scaler, imputer, top_features)

    save_cleaned_top_features(X_scaled_8, y_final)

    print("\nRandom Forest Model Metrics Summary (Top 8 Features):")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")


if __name__ == "__main__":
    main()
