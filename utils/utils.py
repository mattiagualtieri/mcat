import os
import pandas as pd
import h5py
import numpy as np
import torch


def preprocess_labels(input_file, output_file):
    data = pd.read_csv(input_file, sep=';')
    data.set_index('case_id', inplace=True)
    risk_labels, q_bins = pd.qcut(data['overall_survival'], q=4, retbins=True, labels=False)
    cases = 0
    with h5py.File(output_file, 'w') as hdf5_file:
        for case_id in data.index:
            cases += 1
            case_data = data.loc[case_id]
            case_group = hdf5_file.create_group(case_id)
            labels_group = case_group.create_group('labels')
            labels_group.create_dataset('overall_survival_days', data=np.array(case_data['overall_survival']),
                                        dtype='i')
            labels_group.create_dataset('overall_survival_months', data=np.array(case_data['overall_survival'] / 30),
                                        dtype='i')
            labels_group.create_dataset('survival_risk', data=np.array(risk_labels[case_id]), dtype='i')

    print(f'Created labels datasets for {cases} cases')


def preprocess_omics(input_file, signatures_file, output_file):
    omics = pd.read_csv(input_file, sep=';')
    omics.set_index('case_id', inplace=True)
    signatures = pd.read_csv(signatures_file, sep=';')
    cases = 0
    with h5py.File(output_file, 'a') as hdf5_file:
        for case_id in omics.index:
            cases += 1
            omics_data = omics.loc[case_id]
            case_group = hdf5_file[case_id]
            omics_group = case_group.create_group('omics')
            for omics_category in signatures.columns:
                data = []
                for gene in signatures[omics_category].dropna():
                    if gene in omics_data:
                        data.append(omics_data[gene])
                omics_group.create_dataset(omics_category, data=np.array(data), dtype='f')
    print(f'Created omics datasets for {cases} cases')


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


def remove_incomplete_cases(dataset_file):
    with h5py.File(dataset_file, 'r+') as hdf5_file:
        cases = list(hdf5_file.keys())
        removed = 0
        print(f'Total cases in dataset: {len(cases)}')
        for case_id in cases:
            to_remove = False
            case_group = hdf5_file[case_id]
            try:
                case_group['labels']
            except KeyError:
                print(f'"labels" group not found for case {case_id}')
                to_remove = True
            try:
                case_group['omics']
            except KeyError:
                print(f'"omics" group not found for case {case_id}')
                to_remove = True
            try:
                case_group['wsi']
            except KeyError:
                print(f'"wsi" group not found for case {case_id}')
                to_remove = True

            if to_remove:
                print(f'Deleting case {case_id}')
                del hdf5_file[case_id]
                removed += 1

        cases = list(hdf5_file.keys())
        print(f'Total complete cases in dataset: {len(cases)}')


def get_omics_sizes_from_dataset(hdf5_file):
    category_counts = {}
    with h5py.File(hdf5_file, 'r') as f:
        first_case_id = list(f.keys())[0]  # Get the first case ID
        omics_group = f[first_case_id]['omics']  # Access the 'omics' group under the first case ID
        for category in omics_group.keys():
            category_counts.setdefault(category, 0)
            category_counts[category] = len(omics_group[category])  # Count number of values in each category
    sorted_counts = [category_counts[category] for category in sorted(category_counts.keys())]
    return sorted_counts
