import yaml
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from sksurv.metrics import concordance_index_censored

from utils.loss import CrossEntropySurvivalLoss
from utils.utils import *
from models.mcat import MultimodalCoAttentionTransformer
from dataset.dataset import MultimodalDataset


def main():
    with open('config/config.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available')
        device = 'cpu'
    print(f'Running on {device}')

    dataset_file = config['dataset']['dataset_file']

    with (h5py.File(dataset_file, 'r') as hdf5_file):
        # Dataset
        dataset = MultimodalDataset(hdf5_file)
        train_size = config['training']['train_size']
        print(f'Using {int(train_size * 100)}% train, {100 - int(train_size * 100)}% validation')
        train_size = int(train_size * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        # Model
        omics_sizes = get_omics_sizes_from_dataset(dataset_file)
        model = MultimodalCoAttentionTransformer(omic_sizes=omics_sizes)
        model.to(device=device)
        # Loss function
        if config['training']['loss'] == 'ce':
            print('Using CrossEntropyLoss during training')
            loss_function = nn.CrossEntropyLoss()
        elif config['training']['loss'] == 'ces':
            print('Using CrossEntropySurvivalLoss during training')
            loss_function = CrossEntropySurvivalLoss()
        else:
            raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
        # Optimizer
        lr = config['training']['lr']
        weight_decay = config['training']['weight_decay']
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, weight_decay=weight_decay)

        print('Training started')
        model.train()
        grad_acc_step = config['training']['grad_acc_step']
        epochs = config['training']['epochs']
        for epoch in range(epochs):
            train_loss = 0.0
            risk_scores = []
            event_times = []
            for batch_index, (overall_survival_months, survival_risk, omics_data, patches_embeddings) in enumerate(train_loader):
                overall_survival_months = overall_survival_months.to(device)
                survival_risk = survival_risk.to(device)
                patches_embeddings = patches_embeddings.to(device)
                omics_data = [omic_data.to(device) for omic_data in omics_data]
                hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)
                predicted_class = torch.topk(Y, 1, dim=1)[1]

                if config['training']['loss'] == 'ce':
                    loss = loss_function(Y, survival_risk.long())
                elif config['training']['loss'] == 'ces':
                    loss = loss_function(hazards, survs, predicted_class, c=torch.FloatTensor([0]))
                else:
                    raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
                loss_value = loss.item()

                risk = -torch.sum(survs, dim=1).detach().cpu().numpy()
                risk_scores.append(risk.item())
                event_times.append(overall_survival_months.item())

                train_loss += loss_value

                if (batch_index + 1) % 100 == 0:
                    print('batch {}, loss: {:.4f}, label: {}, risk: {:.4f}'.format(
                        batch_index, loss_value, survival_risk.item(), float(risk)))
                loss = loss / grad_acc_step
                loss.backward()

                if (batch_index + 1) % grad_acc_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # calculate loss and error for epoch
            train_loss /= len(train_loader)
            c_index = concordance_index_censored(np.ones((len(train_loader))).astype(bool), np.array(event_times),
                                                 np.array(risk_scores))[0]
            print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, c_index))

        print('Evaluation started')
        model.eval()
        val_loss = 0.0
        risk_scores = []
        event_times = []
        for batch_index, (overall_survival_months, survival_risk, omics_data, patches_embeddings) in enumerate(val_loader):
            overall_survival_months = overall_survival_months.to(device)
            survival_risk = survival_risk.to(device)
            patches_embeddings = patches_embeddings.to(device)
            omics_data = [omic_data.to(device) for omic_data in omics_data]
            with torch.no_grad():
                hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)
            predicted_class = torch.topk(Y, 1, dim=1)[1]

            if config['training']['loss'] == 'ce':
                loss = loss_function(Y, survival_risk.long())
            elif config['training']['loss'] == 'ces':
                loss = loss_function(hazards, survs, predicted_class, c=torch.FloatTensor([0]))
            else:
                raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
            loss_value = loss.item()

            risk = -torch.sum(survs, dim=1).cpu().numpy()
            risk_scores.append(risk.item())
            event_times.append(overall_survival_months.item())

            val_loss += loss_value

        # calculate loss and error
        val_loss /= len(val_loader)
        c_index = concordance_index_censored(np.ones((len(val_loader))).astype(bool), np.array(event_times),
                                             np.array(risk_scores))[0]
        print('Validation: val_loss: {:.4f}, val_c_index: {:.4f}'.format(val_loss, c_index))


if __name__ == '__main__':
    main()
