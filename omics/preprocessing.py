import yaml
import pandas as pd
import h5py
import numpy as np


def preprocess_omics(input_file, signatures_file, output_file):
    omics = pd.read_csv(input_file, sep=';')
    omics.set_index('case_id', inplace=True)
    signatures = pd.read_csv(signatures_file, sep=';')
    cases = 0
    with h5py.File(output_file, "w") as hdf5_file:
        for case_id in omics.index:
            cases += 1
            omics_data = omics.loc[case_id]
            case_group = hdf5_file.create_group(case_id)
            omics_group = case_group.create_group('omics')
            for omics_category in signatures.columns:
                data = []
                for gene in signatures[omics_category].dropna():
                    if gene in omics_data:
                        data.append(omics_data[gene])
                category_dataset = omics_group.create_dataset(omics_category, data=np.array(data), dtype='f')
    print(f'Created omics datasets for {cases} cases')
