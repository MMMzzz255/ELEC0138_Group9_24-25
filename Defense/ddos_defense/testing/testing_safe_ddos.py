import pandas as pd
import numpy as np
import os

# Define top 8 features + Class (note: the Class column is only used for filtering)
TOP_8_FEATURES = [
    "Bwd Packets/s",
    "Down/Up Ratio",
    "Packet Length Min",
    "Fwd Packet Length Min",
    "Avg Bwd Segment Size",
    "URG Flag Count",
    "Bwd Packet Length Max",
    "ACK Flag Count",

]


def sample_and_clean_top_features(input_path, output_path, sample_fraction=0.1, random_state=42):
    """
    Load the dataset, filter for safe records (where Class == 0), select the top 8 features,
    clean the data by replacing infinite values and dropping NaNs, sample a fraction of the data,
    and save the resulting DataFrame.

    :param input_path: Path to the input CSV file.
    :param output_path: Path to save the cleaned and sampled CSV file.
    :param sample_fraction: Fraction of the dataset to sample.
    :param random_state: Random state for reproducibility.
    """
    # Load data
    df = pd.read_csv(input_path)

    # Filter for safe records (assumes safe records are labeled with 0)
    df = df[df["Class"] == "Benign"]

    # Select only the top 8 features (dropping the Class column afterwards)
    df = df[TOP_8_FEATURES]

    # Replace inf/-inf with NaN and drop rows with NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Take a random sample of the cleaned data
    df_sampled = df.sample(frac=sample_fraction, random_state=random_state)

    # Ensure the output directory exists and save the sampled data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sampled.to_csv(output_path, index=False)
    print(f"Cleaned, safe-only, and sampled data saved to: {output_path}")


def main():
    """
    Main function to sample and clean the dataset.
    """
    input_path = "../data/ddos_data.csv"
    output_path = "../../../../../Desktop/testing/ddos_test_data_safe.csv"
    sample_fraction = 0.1

    sample_and_clean_top_features(input_path, output_path, sample_fraction)


if __name__ == "__main__":
    main()
