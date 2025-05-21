import pandas as pd
import os

def prepare_dataset(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Drop rows with missing values
    df = df.dropna()

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Save the cleaned dataset to a new CSV file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    for file in os.listdir('data'):
        if file.endswith('.csv') and 'cleaned' not in file:
            input_file = os.path.join('data', file)
            output_file = os.path.join('data', f'cleaned_{file}')
            prepare_dataset(input_file, output_file)
