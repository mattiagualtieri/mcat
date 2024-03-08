import yaml
import pandas as pd


def preprocess():
    with open('../config/omics-preprocessing.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    signatures_file = config['signatures']
    input_file = config['input']

    omics = pd.read_csv(input_file, sep=';')
    signatures = pd.read_csv(signatures_file, sep=';')
    output_dir = config['output_dir']
    category_index = 1
    for category in signatures.columns:
        # Initialize a DataFrame for the current category
        print(f'Generating "{category}" data ({category_index}.csv)')
        category_df = pd.DataFrame(columns=['case_id'] + list(signatures[category].dropna()))
        category_df = category_df.dropna(axis=1)
        for entry in category_df.columns:
            if entry in omics:
                category_df[entry] = omics[entry]

        category_df.to_csv(f'{output_dir}/{category_index}.csv', index=False, sep=';')
        category_index += 1

    return output_dir


if __name__ == '__main__':
    output_dir = preprocess()
    print(f'Created file(s) in {output_dir}')
