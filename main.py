import yaml
import os
import torch
import h5py

from torch.utils.data import DataLoader

from labels.preprocessing import preprocess_labels
from omics.preprocessing import preprocess_omics
from wsi.preprocessing import preprocess_patch_embeddings
from dataset.dataset import MultimodalDataset


def main():
    with open('config/config.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')

    dataset_file = config['dataset_file']
    if os.path.exists(dataset_file):
        print(f'Skipping {dataset_file} creation: already exists')
    else:
        raw_input = config['raw_input']

        preprocess_labels(raw_input, dataset_file)
        print(f'Created dataset {dataset_file}')

        omics_signatures = config['omics_signatures']
        preprocess_omics(raw_input, omics_signatures, dataset_file)
        print(f'Added omics data to dataset {dataset_file}')

        # here there should be all the part in which:
        # - take each WSI slide
        # - divide it into M patches (256x256)
        # - embed each patch with a ResNet50 (pretrained), so we obtain a Mx1024 matrix
        print('Note: skipping all patches creation and embedding part...')
        # - The matrix is created into a .pt file, that we must read and add to the dataset
        patch_emb_dir = config['patch_emb_dir']
        preprocess_patch_embeddings(patch_emb_dir, dataset_file)
        print(f'Added patch embeddings to dataset {dataset_file}')

    with h5py.File(dataset_file, 'r') as hdf5_file:
        dataset = MultimodalDataset(hdf5_file)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # Iterate over batches of data from your dataset
        for batch_index, (overall_survival, survival_risk, omics_data, patches_embeddings) in enumerate(dataloader):
            # Process the data as needed
            # print("Omics Data:", omics_data)
            print("Overall Survival:", overall_survival)
            print("Survival Risk:", survival_risk)


if __name__ == '__main__':
    main()
