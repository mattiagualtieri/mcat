import yaml

from torch.utils.data import DataLoader

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
    if os.path.exists(dataset_file):
        print(f'Skipping {dataset_file} creation: already exists')
    else:
        raw_input = config['inputs']['raw_input']

        preprocess_labels(raw_input, dataset_file)
        print(f'Created dataset {dataset_file}')

        omics_signatures = config['inputs']['omics_signatures']
        preprocess_omics(raw_input, omics_signatures, dataset_file)
        print(f'Added omics data to dataset {dataset_file}')

        # here there should be all the part in which:
        # - take each WSI slide
        # - divide it into M patches (256x256)
        # - embed each patch with a ResNet50 (pretrained), so we obtain a Mx1024 matrix
        print('Note: skipping all patches creation and embedding part...')
        # - The matrix is created into a .pt file, that we must read and add to the dataset
        patch_emb_dir = config['inputs']['patch_emb_dir']
        preprocess_patch_embeddings(patch_emb_dir, dataset_file)
        print(f'Added patch embeddings to dataset {dataset_file}')

    with h5py.File(dataset_file, 'r') as hdf5_file:
        # Dataset
        dataset = MultimodalDataset(hdf5_file)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        # Model
        omics_sizes = get_omics_sizes_from_dataset(dataset_file)
        model = MultimodalCoAttentionTransformer(omic_sizes=omics_sizes)
        # Loss function
        loss_function = CrossEntropySurvivalLoss()
        # Optimizer
        lr = config['training']['lr']
        weight_decay = config['training']['weight_decay']
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, weight_decay=weight_decay)

        model.train()
        train_loss_surv = 0.0
        grad_acc_step = config['training']['grad_acc_step']
        epochs = config['training']['epochs']
        for epoch in range(epochs):
            for batch_index, (_, survival_risk, omics_data, patches_embeddings) in enumerate(loader):
                hazards, S, Y_hat, attention_scores = model(wsi=patches_embeddings, omics=omics_data)
                loss = loss_function(hazards, S, Y_hat, c=torch.FloatTensor([1]))
                loss_value = loss.item()

                risk = -torch.sum(S, dim=1).detach().cpu().numpy()

                train_loss_surv += loss_value

                if (batch_index + 1) % 100 == 0:
                    print('batch {}, loss: {:.4f}, label: {}, risk: {:.4f}, bag_size:'.format(
                        batch_index, loss_value, survival_risk.item(), float(risk)))
                loss = loss / grad_acc_step
                loss.backward()

                if (batch_index + 1) % grad_acc_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # calculate loss and error for epoch
            train_loss_surv /= len(loader)
            # c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
            print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
                epoch, train_loss_surv, 0, 0))


if __name__ == '__main__':
    main()
