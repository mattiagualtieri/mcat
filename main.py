import torch.cuda
import yaml
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from sksurv.metrics import concordance_index_censored

from utils.loss import CrossEntropySurvivalLoss
from utils.utils import *
from models.mcat import MultimodalCoAttentionTransformer
from dataset.dataset import MultimodalDataset


def train(epoch, config, device, train_loader, model, loss_function, grad_acc_step, optimizer):
    model.train()
    train_loss = 0.0
    risk_scores = []
    event_times = []
    for batch_index, (overall_survival_months, survival_risk, omics_data, patches_embeddings) in enumerate(
            train_loader):
        overall_survival_months = overall_survival_months.to(device)
        survival_risk = survival_risk.to(device)
        survival_risk = survival_risk.unsqueeze(0).to(torch.int64)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)

        if config['training']['loss'] == 'ce':
            loss = loss_function(Y, survival_risk.long())
        elif config['training']['loss'] == 'ces':
            loss = loss_function(hazards, survs, survival_risk, c=torch.FloatTensor([0]).to(device))
        else:
            raise RuntimeError(f'Loss "{config["training"]["loss"]}" not implemented')
        loss_value = loss.item()

        risk = -torch.sum(survs, dim=1).detach().cpu().numpy()
        risk_scores.append(risk.item())
        event_times.append(overall_survival_months.item())

        train_loss += loss_value

        if (batch_index + 1) % 32 == 0:
            print('\tbatch: {}, loss: {:.4f}, label: {}, risk: {:.4f}'.format(
                batch_index, loss_value, survival_risk.item(), float(risk.item())))
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


def validate(epoch, config, device, val_loader, model, loss_function):
    model.eval()
    val_loss = 0.0
    risk_scores = []
    event_times = []
    for batch_index, (overall_survival_months, survival_risk, omics_data, patches_embeddings) in enumerate(val_loader):
        overall_survival_months = overall_survival_months.to(device)
        survival_risk = survival_risk.to(device)
        survival_risk = survival_risk.unsqueeze(0).to(torch.int64)
        patches_embeddings = patches_embeddings.to(device)
        omics_data = [omic_data.to(device) for omic_data in omics_data]
        with torch.no_grad():
            hazards, survs, Y, attention_scores = model(wsi=patches_embeddings, omics=omics_data)

        if config['training']['loss'] == 'ce':
            loss = loss_function(Y, survival_risk.long())
        elif config['training']['loss'] == 'ces':
            loss = loss_function(hazards, survs, survival_risk, c=torch.FloatTensor([0]).to(device))
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
    print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, c_index))


def main():
    with open('config/config.yaml') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available')
        device = 'cpu'
    elif device == 'cuda' and torch.cuda.is_available():
        print('CUDA is available!')
        print(f'Device count: {torch.cuda.device_count()}')
        current_device_index = torch.cuda.current_device()
        print(f'Current device index: {current_device_index}')
        print(f'Current device: {torch.cuda.get_device_name(current_device_index)}')
    print(f'Running on {device.upper()}')

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

        print('-- Started training')
        model.train()
        grad_acc_step = config['training']['grad_acc_step']
        epochs = config['training']['epochs']
        for epoch in range(epochs):
            train(epoch, config, device, train_loader, model, loss_function, grad_acc_step, optimizer)
            validate(epoch, config, device, val_loader, model, loss_function)

        validate('final validation', config, device, val_loader, model, loss_function)


if __name__ == '__main__':
    main()
