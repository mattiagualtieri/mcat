import os
import h5py
import numpy as np
import torch


def preprocess_patch_embeddings(emb_dir, output_file):
    with h5py.File(output_file, 'a') as hdf5_file:
        cases = 0
        for file in os.listdir(emb_dir):
            if file.endswith('.h5'):
                filename = os.path.basename(file)
                case_id = os.path.splitext(filename)[0]
                case_group = hdf5_file[case_id]
                wsi_group = case_group.create_group('wsi')
                with h5py.File(os.path.join(emb_dir, file), 'r') as emb_hdf5_file:
                    embeddings = torch.tensor(emb_hdf5_file['features'][()])
                    wsi_group.create_dataset('patches', data=np.array(embeddings), dtype='f')
                    cases += 1
    print(f'Created WSI datasets for {cases} cases')