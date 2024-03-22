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

        labels_group = case_group['labels']
        overall_survival = torch.tensor(labels_group['overall_survival'][()])
        survival_risk = torch.tensor(labels_group['survival_risk'][()])

        omics_group = case_group['omics']
        omics_data = [torch.tensor(omics_group[dataset][:]) for dataset in omics_group]

        wsi_group = case_group['wsi']
        patches_embeddings = torch.tensor(wsi_group['patches'][()])

        return overall_survival, survival_risk, omics_data, patches_embeddings
