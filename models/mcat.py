import numpy as np
import torch
import torch.nn as nn

# https://github.com/mahmoodlab/MCAT/blob/master/Model%20Computation%20%2B%20Complexity%20Overview.ipynb


class MultimodalCoAttentionTransformer(nn.Module):
    def __init__(self, omic_sizes: [], n_classes: int = 4):
        super(MultimodalCoAttentionTransformer, self).__init__()
        self.n_classes = n_classes
        self.d_k = 256

        # H
        fc = nn.Sequential(
            nn.Linear(1024, self.d_k),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.H = fc

        # G
        omic_encoders = []
        for omic_size in omic_sizes:
            fc = nn.Sequential(
                nn.Sequential(
                    nn.Linear(omic_size, 256),
                    nn.ELU(),
                    nn.AlphaDropout(p=0.25, inplace=False)),
                nn.Sequential(
                    nn.Linear(256, self.d_k),
                    nn.ELU(),
                    nn.AlphaDropout(p=0.25, inplace=False))
            )
            omic_encoders.append(fc)
        self.G = nn.ModuleList(omic_encoders)

        # Genomic-Guided Co-Attention
        self.co_attention = nn.MultiheadAttention(embed_dim=self.d_k, num_heads=1)

    def forward(self, wsi, omics):
        # WSI embeddings are fed through an FC layer (Mxd_k)
        H_bag = self.H(wsi).unsqueeze(1)

        # Each omic signature goes through its own FC layer
        G_omic = [self.G[index].forward(omic) for index, omic in enumerate(omics)]
        # Omic embeddings are stacked to be used in Co-Attention (Nxd_k)
        G_bag = torch.stack(G_omic).unsqueeze(1)

        # Co-Attention results
        # Genomic-Guided WSI-level Embeddings (Nxd_k)
        # Co-Attention Matrix (NxM)
        H_coattn, A_coattn = self.co_attention(query=G_bag, key=H_bag, value=H_bag)


def test():
    print('Testing MultimodalCoAttentionTransformer...')

    wsi = torch.randn((15231, 1024))
    omics = [torch.randn(dim) for dim in [100, 200, 300, 400, 500, 600]]
    omic_sizes = [omic.size()[0] for omic in omics]
    model = MultimodalCoAttentionTransformer(omic_sizes=omic_sizes)
    model(wsi, omics)

    print('Forward successful')


if __name__ == '__main__':
    test()
