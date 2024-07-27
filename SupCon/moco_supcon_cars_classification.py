from torchvision import transforms
from tqdm.contrib.telegram import trange, tqdm

# from custom files
from dataset import CompCarsImageFolder, WrapperDataset, TwoCropTransform
from resnet import ResNet, resnet_cfg
from moco_supcon import MoCo, train
from losses import SupLoss
from utils import *

# DEFAULT PATHS
root = '/home/ubuntu/data/image/'
train_file = '/home/ubuntu/data/train_test_split/classification/train.txt'
test_file = '/home/ubuntu/data/train_test_split/classification/test.txt'

resnet_type = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50'

params = {
    # Training Params (inspired from original resnet paper: https://arxiv.org/pdf/1512.03385)
    'epoch_num': 50,                    # number of epochs
    'lr': 1e-1,                         # (initial) Learning Rate
    'weight_decay': 1e-4,               # L2 Penalty
    'batch_size': 256,                  # batch size (depends on hardware)
    'momentum': 0.9,

    'hierarchy': 0,                     # Choose 0 for manufacturer classification, 1 for model classification
    'val_split': 10000,                 # (float) Fraction of validation holdout / (int) Absolute number of data points in holdout

    'resnet': resnet_cfg[resnet_type],  # ResNet configuration

    'use_train_test_split': False,      # True: use prepared split, False: use total dataset

    'use_amp': False,                   # Automatic Mixed Precision (AMP) for faster training on NVIDIA GPUs

    # TO DO
    'moco_dim': 128,
    'moco_k': 1024,                     # queue size (carmaker: 1024, car model: 8192)
    'moco_m': 0.999,
    'moco_t': 0.2,                      # temperature parameter
    'mlp': True
}


def set_device():
    if torch.cuda.is_available():
        params["device"] = torch.device("cuda")  # option for NVIDIA GPUs
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        params["device"] = torch.device("mps")  # option for Mac M-series chips (GPUs)
    else:
        params["device"] = torch.device("cpu")  # default option if none of the above devices are available

    print("Device: {}".format(params["device"]))


def set_loader() -> dict[str, DataLoader]:
    # Load full dataset
    # hierarchy=0 -> manufacturer classification; hierarchy=1 -> model classification
    total_set = CompCarsImageFolder(root, hierarchy=params['hierarchy'])
    datasets = train_val_dataset(total_set, val_split=params['val_split'])

    # default (computed statistic on whole dataset)
    mean, std = [0.483, 0.471, 0.463], [0.297, 0.296, 0.302]

    # Compute mean and std for training dataset
    # train_mean, train_std = compute_mean_std_from_dataset(datasets['train'])
    # print(f"Training dataset mean: {train_mean}")
    # print(f"Training dataset std: {train_std}")
    train_mean, train_std = mean, std

    # Compute mean and std for validation dataset
    # val_mean, val_std = compute_mean_std_from_dataset(datasets['val'])
    # print(f"Validation dataset mean: {val_mean}")
    # print(f"Validation dataset std: {val_std}")
    val_mean, val_std = mean, std

    # inspired from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomChoice([
                transforms.Resize(256),
                transforms.Resize(224),
                transforms.Resize(320)
            ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(val_mean, val_std)
        ])
    }

    wrapped_datasets = {
        'train': WrapperDataset(datasets['train'], transform=TwoCropTransform(data_transforms['train'])),
        'val': WrapperDataset(datasets['val'], transform=data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(wrapped_datasets['train'], batch_size=params['batch_size'], shuffle=True, num_workers=4, drop_last=True),
        'val': DataLoader(wrapped_datasets['val'], batch_size=params['batch_size'], shuffle=False, num_workers=4)
    }

    print(f"Training dataset size: {len(wrapped_datasets['train'])}")
    print(f"Validation dataset size: {len(wrapped_datasets['val'])}")

    x, y, _ = next(iter(dataloaders['train']))
    print(f"Batch of training images shape: {x[0].shape}")
    print(f"Batch of training labels shape: {y.shape}")

    return dataloaders


def set_model(dataloaders: dict[str, Any]) -> tuple[MoCo, SupLoss]:
    # create model
    encoder = ResNet(params['resnet']['block'], params['resnet']['layers'], num_classes=params['moco_dim'])

    model = MoCo(encoder, 
                 dim=params['moco_dim'], 
                 K=params['moco_k'], 
                 m=params['moco_m'], 
                 T=params['moco_t'], 
                 mlp=params['mlp'])

    model.to(params['device'])

    for _, labels, index in tqdm(dataloaders['train'], miniters=1):
        labels = labels.to(params['device'])
        index = index.to(params['device'])
        model._init_label_information(index, labels)

    model._show_label_information()

    # Loss and Optimizer
    criterion = SupLoss(temperature=params['moco_t'], K=params['moco_k']).to(params['device'])

    return model, criterion


def set_optimizer(model) -> tuple[torch.optim.SGD, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        momentum=params['momentum']
    )

    # LR scheduler (using Loss as Validation metric)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, min_lr=1e-4, patience=3,
                                                           threshold=1e-3)
    return optimizer, scheduler


def save_model(model: MoCo, optimizer: torch.optim, epoch: int, train_losses: list):
    MODEL_PATH = './trained_models/pretrained_moco_weights_car_'

    if params['hierarchy'] == 0:
        MODEL_PATH += 'makers_'
    else:
        MODEL_PATH += 'models_'
        
    MODEL_PATH += str(params['batch_size']) + '.pth'

    torch.save({
        'params': params,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer,
        'epoch': epoch,
        'train_losses': train_losses
        }, MODEL_PATH)


def main():
    set_device()

    dataloaders = set_loader()

    model, criterion = set_model(dataloaders)

    optimizer, scheduler = set_optimizer(model)

    # Save performance metrics
    train_losses = list()

    CHECKPOINT_PATH = './training_checkpoints/checkpoint.pth'
    START_FROM_CHECKPOINT = False  # set to TRUE to start from checkpoint
    start_epoch = 0

    if START_FROM_CHECKPOINT:
        checkpoint = torch.load(CHECKPOINT_PATH)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_losses = checkpoint['train_losses']

    # Just some fancy progress bars # FIXME: inside epoch progress bar not working reliably for me
    pbar_epoch = trange(start_epoch, params["epoch_num"], initial=start_epoch, total=params["epoch_num"],
                        desc="Training", unit="epoch", position=0, leave=True,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]",
                        token="7201508620:AAFKipOQ7_Xdcgid1xDf60fCCkJuKAcPVBw", chat_id="-4239730104")
    pbar_inside_epoch = trange(0, len(dataloaders['train']), desc="Training per epoch", position=1, leave=False,
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]",
                               token="7201508620:AAFKipOQ7_Xdcgid1xDf60fCCkJuKAcPVBw", chat_id="-4239730104")

    for epoch in pbar_epoch:
        pbar_inside_epoch.reset()

        # Training
        train_results = train(dataloaders['train'], model, epoch, criterion, optimizer, params["device"],
                              pbar=pbar_inside_epoch)
        train_losses.append(train_results)

        # ReduceLROnPlateau scheduler (reduce LR by 10 when loss does not improve)
        scheduler.step(train_results)
        print("\nCurrent Learning Rate: ", round(scheduler.get_last_lr()[0], 4), "\n")

        # Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
        }, CHECKPOINT_PATH)

    pbar_inside_epoch.close()

    save_model(model, optimizer, params["epoch_num"], train_losses)


if __name__ == '__main__':
    main()
