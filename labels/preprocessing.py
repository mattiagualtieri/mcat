import pandas as pd
import h5py
import numpy as np


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
            labels_group.create_dataset('overall_survival', data=np.array(case_data['overall_survival']), dtype='i')
            labels_group.create_dataset('survival_risk', data=np.array(risk_labels[case_id]), dtype='i')

    print(f'Created labels datasets for {cases} cases')
