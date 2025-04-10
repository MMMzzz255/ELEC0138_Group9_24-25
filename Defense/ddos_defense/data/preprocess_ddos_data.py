import pandas as pd
import os


def save_preprocessed_data(X, output_path="../data/preprocessed_ddos_data.csv"):
    """
    Save the preprocessed data to a CSV file.
    :param X: DataFrame to save
    :param output_path: Path to save the preprocessed data
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X.to_csv(output_path, index=False)


def main():
    """
    Main function to preprocess the DDoS data.
    """
    df = pd.read_csv("ddos_data.csv")
    df = df.sample(frac=1 / 3, random_state=42)
    df.to_csv("../data/preprocessed_ddos_data.csv", index=False)
    print("Preprocessing complete. 1/3 of the data saved to ../data/preprocessed_data.csv")


if __name__ == "__main__":
    main()
