import yaml
import os
import torch
import h5py

from torch.utils.data import DataLoader

from dataset.dataset import MultimodalDataset
from labels.preprocessing import preprocess_labels
from omics.preprocessing import preprocess_omics


def main():
    with open('config/config.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')

    raw_input = config['raw_input']
    dataset_file = config['dataset_file']
    if os.path.exists(dataset_file):
        print(f'Skipping {dataset_file} creation: already exists')
    else:
        preprocess_labels(raw_input, dataset_file)
        print(f'Created dataset {dataset_file}')

        omics_signatures = config['omics_signatures']
        preprocess_omics(raw_input, omics_signatures, dataset_file)
        print(f'Added omics data to dataset {dataset_file}')

    with h5py.File(dataset_file, 'r') as hdf5_file:
        dataset = MultimodalDataset(hdf5_file)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Iterate over batches of data from your dataset
        for omics_data, overall_survival, survival_risk in dataloader:
            # Process the data as needed
            # print("Omics Data:", omics_data)
            print("Overall Survival:", overall_survival)
            print("Survival Risk:", survival_risk)


if __name__ == '__main__':
    main()
