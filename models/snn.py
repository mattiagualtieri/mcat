import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SNN(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 4):
        super(SNN, self).__init__()
        self.n_classes = n_classes

        self.fc = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ELU(),
                nn.AlphaDropout(p=0.25, inplace=False)),
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ELU(),
                nn.AlphaDropout(p=0.25, inplace=False)),
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ELU(),
                nn.AlphaDropout(p=0.25, inplace=False)),
            nn.Sequential(
                nn.Linear(256, 256),
                nn.ELU(),
                nn.AlphaDropout(p=0.25, inplace=False))
        )
        self.classifier = nn.Linear(256, n_classes)
        self.init_weights(self)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.fc_omic(x)
        logits = self.classifier(features).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, Y_hat

    def relocate(self):
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
            else:
                self.fc_omic = self.fc_omic.to(device)


            self.classifier = self.classifier.to(device)

    @staticmethod
    def init_weights(module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                standard_deviation = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.normal_(0, standard_deviation)
                m.bias.data.zero_()
