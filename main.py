import yaml
import torch

from omics.preprocessing import preprocess_omics


def main():
    with open('config/config.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')

    omics_raw_input = config['omics_raw_input']
    omics_signatures = config['omics_signatures']
    omics_processed_output_dir = config['omics_processed_output_dir']
    preprocess_omics(omics_raw_input, omics_signatures, omics_processed_output_dir)
    omics_input_dir = omics_processed_output_dir


if __name__ == '__main__':
    main()
