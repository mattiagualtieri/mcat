import yaml
import pandas as pd


def preprocess_omics(input_file, signatures_file, output_dir):
    omics = pd.read_csv(input_file, sep=';')
    signatures = pd.read_csv(signatures_file, sep=';')
    for category in signatures.columns:
        # Initialize a DataFrame for the current category
        print(f'Generating {category}.csv')
        category_df = pd.DataFrame(columns=['case_id'] + list(signatures[category].dropna()))
        category_df = category_df.dropna(axis=1)
        for entry in category_df.columns:
            if entry in omics:
                category_df[entry] = omics[entry]

        category_df.to_csv(f'{output_dir}/{category}.csv', index=False, sep=';')
