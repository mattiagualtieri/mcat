import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.case_ids = list(self.hdf5_file.keys())

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        case_group = self.hdf5_file[case_id]

        omics_group = case_group['omics']
        omics_data = [torch.tensor(omics_group[dataset][:]) for dataset in omics_group]

        labels_group = case_group['labels']
        overall_survival = torch.tensor(labels_group['overall_survival'][()])
        survival_risk = torch.tensor(labels_group['survival_risk'][()])

        return omics_data, overall_survival, survival_risk
