import yaml
import torch

from labels.preprocessing import preprocess_labels
from omics.preprocessing import preprocess_omics


def main():
    with open('config/config.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')

    raw_input = config['raw_input']
    dataset_file = config['dataset_file']
    preprocess_labels(raw_input, dataset_file)
    print(f'Created dataset {dataset_file}')

    omics_signatures = config['omics_signatures']
    preprocess_omics(raw_input, omics_signatures, dataset_file)
    print(f'Added omics data to dataset {dataset_file}')


if __name__ == '__main__':
    main()
