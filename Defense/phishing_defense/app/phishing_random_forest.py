import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score


def load_and_clean_data(path):
    """
    Load and clean the dataset.
    :param path: Path to the dataset
    :return: Cleaned DataFrame
    """
    df = pd.read_csv(path)
    df.replace([-np.inf, np.inf, -1], np.nan, inplace=True)
    df.dropna(subset=["URL_Type_obf_Type"], inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, df.nunique() > 1]
    return df


def encode_target_and_split(df):
    """
    Encode the target variable and split the dataset into features and target.
    :param df: DataFrame containing the dataset
    :return: Tuple of features DataFrame and target Series
    """
    y = df['URL_Type_obf_Type'].apply(lambda x: 1 if x != 'benign' else 0)
    X = df.drop(columns=['URL_Type_obf_Type'])
    return X, y


def handle_extreme_values(X):
    """
    Handle extreme values in the dataset by replacing them with NaN.
    :param X: Features DataFrame
    :return: DataFrame with cleaned data
    """
    return X.applymap(lambda x: np.nan if x is not None and abs(x) > 1e10 else x)


def impute_missing_values(X, y):
    """
    Impute missing values in the dataset using mean strategy.
    :param X: Features DataFrame
    :param y: Target Series
    :return: Tuple of imputed features DataFrame, target Series, and imputer object
    """
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_imputed.dropna(inplace=True)
    y_cleaned = y.loc[X_imputed.index].copy()
    return X_imputed, y_cleaned, imputer


def get_top_features(X, y, k=20):
    """
    Get the top k features based on feature importance from a Random Forest model.
    :param X: Features DataFrame
    :param y: Target Series
    :param k: Number of top features to select
    :return: List of top feature names
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(k).index.tolist()
    return top_features


def extract_and_impute_top_features(X, top_feature_names):
    """
    Extract the top features and impute missing values.
    :param X: Features DataFrame
    :param top_feature_names: List of top feature names
    :return: Tuple of imputed features DataFrame and imputer object
    """
    X_top = X[top_feature_names]
    imputer = SimpleImputer(strategy='mean')
    X_top_imputed = pd.DataFrame(imputer.fit_transform(X_top), columns=top_feature_names)
    X_top_imputed.dropna(inplace=True)
    return X_top_imputed, imputer


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model on the training data.
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained Random Forest model
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy',
                               verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_


def save_artifacts(model, imputer, top_feature_names, output_dir="../model"):
    """
    Save the trained model, imputer, and top feature names to disk.
    :param model: Trained Random Forest model
    :param imputer: Imputer object
    :param top_feature_names: List of top feature names
    :param output_dir: Directory to save the artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "rf_model.pkl"))
    joblib.dump(imputer, os.path.join(output_dir, "rf_imputer.pkl"))
    joblib.dump(top_feature_names, os.path.join(output_dir, "rf_top_features.pkl"))


def save_encoded_dataset(X, y, output_path="../data/processed_phishing_data.csv"):
    """
    Save the cleaned and encoded dataset to a CSV file.
    :param X: Features DataFrame
    :param y: Target Series
    :param output_path: Path to save the encoded dataset
    """
    df_encoded = X.copy()
    df_encoded['Target'] = y.values
    df_encoded.to_csv(output_path, index=False)


def save_confusion_matrix(cm, labels, filename="../result/confusion_matrix_rf.png"):
    """
    Save the confusion matrix as an image file.
    :param cm: Confusion matrix
    :param labels: List of class labels
    :param filename: Path to save the confusion matrix image
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


def save_metrics(y_true, y_pred, filename="../result/metrics_rf.csv"):
    """
    Save evaluation metrics to a CSV file.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param filename: Path to save the metrics CSV
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
    metrics_df = pd.DataFrame([metrics])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    metrics_df.to_csv(filename, index=False)

    print("\n===== Random Forest Model Metrics Summary =====")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")


def main():
    # Load and clean the data
    df = load_and_clean_data("../data/phishing_data.csv")

    # Encode the target variable and split the dataset
    X, y = encode_target_and_split(df)

    # Handle extreme values
    X = handle_extreme_values(X)

    # Impute full dataset
    X_full_imputed, y_full, initial_imputer = impute_missing_values(X, y)

    # Get top features using full data
    top_feature_names = get_top_features(X_full_imputed, y_full)

    # Extract and impute only top features
    X_top_imputed, final_imputer = extract_and_impute_top_features(X[top_feature_names], top_feature_names)

    # Align labels again
    y_top = y.loc[X_top_imputed.index]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_top_imputed, y_top, test_size=0.2, random_state=42)

    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate and save confusion matrix and metrics
    y_pred = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    save_confusion_matrix(cm, labels=["Benign", "Phishing"])
    save_metrics(y_test, y_pred)

    # Save model and preprocessing artifacts
    save_artifacts(rf_model, final_imputer, top_feature_names)

    # Save cleaned and encoded top feature dataset
    save_encoded_dataset(X_top_imputed, y_top)


if __name__ == "__main__":
    main()
