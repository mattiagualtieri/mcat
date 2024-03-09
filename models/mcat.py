import numpy as np
import torch
import torch.nn as nn


class MultimodalCoAttentionTransformer(nn.Module):
    def __init__(self, omic_sizes: [], n_classes: int = 4):
        super(MultimodalCoAttentionTransformer, self).__init__()
        self.n_classes = n_classes

        omic_encoders = []
        for omic_size in omic_sizes:
            fc = nn.Sequential(
                nn.Sequential(
                    nn.Linear(omic_size, 256),
                    nn.ELU(),
                    nn.AlphaDropout(p=0.25, inplace=False)),
                nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ELU(),
                    nn.AlphaDropout(p=0.25, inplace=False))
            )
            omic_encoders.append(fc)

        self.G = nn.ModuleList(omic_encoders)

    def forward(self, omics):
        # Each omic signature goes through its own FC layer
        h_omic = [self.G[index].forward(omic) for index, omic in enumerate(omics)]
        # Omic embeddings are stacked (to be used in co-attention)
        h_omic_bag = torch.stack(h_omic).unsqueeze(1)


def test():
    print('Testing MultimodalCoAttentionTransformer...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Running on {device}')
    n_samples = 10
    omics = [
        torch.tensor(np.random.rand(n_samples, 20), dtype=torch.float, device=device),
        torch.tensor(np.random.rand(n_samples, 10), dtype=torch.float, device=device),
        torch.tensor(np.random.rand(n_samples, 30), dtype=torch.float, device=device),
    ]
    omic_sizes = [omic.size()[1] for omic in omics]
    model = MultimodalCoAttentionTransformer(omic_sizes=omic_sizes)
    model(omics)


if __name__ == '__main__':
    test()
