import yaml

from utils.utils import *


def create_dataset():
    with open('config/config.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available')
        device = 'cpu'
    print(f'Running on {device}')

    dataset_file = config['dataset']['dataset_file']
    if os.path.exists(dataset_file):
        print(f'Dataset {dataset_file} already exists, removing')
        os.remove(dataset_file)

    raw_input = config['inputs']['raw_input']

    preprocess_labels(raw_input, dataset_file)
    print(f'Created dataset {dataset_file}')

    omics_signatures = config['inputs']['omics_signatures']
    preprocess_omics(raw_input, omics_signatures, dataset_file)
    print(f'Added omics data to dataset {dataset_file}')

    # here there should be all the part in which:
    # - take each WSI slide
    # - divide it into M patches (256x256)
    # - embed each patch with a ResNet50 (pretrained), so we obtain a Mx1024 matrix
    print('Note: skipping all patches creation and embedding part...')
    # - The matrix is created into a .pt file, that we must read and add to the dataset
    patch_emb_dir = config['inputs']['patch_emb_dir']
    preprocess_patch_embeddings(patch_emb_dir, dataset_file)
    print(f'Added patch embeddings to dataset {dataset_file}')

    print('Removing incomplete samples...')
    remove_incomplete_cases(dataset_file)

    print(f'Dataset created successfully')


if __name__ == '__main__':
    create_dataset()
