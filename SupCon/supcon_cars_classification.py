from torchvision.transforms import v2
from tqdm import trange

# from custom files
from dataset import CompCarsImageFolder, WrapperDataset, TwoCropTransform
from models import resnet_cfg, SupConResNet
from models import train
from utils import *
from losses import SupConLoss

# EDO'S PATHS
root = '/Volumes/EDO/NNDL/CompCars dataset/data/image/'
train_file = '/Volumes/EDO/NNDL/CompCars dataset/data/train_test_split/classification/train.txt'
test_file = '/Volumes/EDO/NNDL/CompCars dataset/data/train_test_split/classification/test.txt'

# DEFAULT PATHS
# root = '../cars_data/data/image/'
# train_file = '../cars_data/data/train_test_split/classification/train.txt'
# test_file = '../cars_data/data/train_test_split/classification/test.txt'

resnet_type = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50'

params = {
    # Training Params (inspired from original resnet paper: https://arxiv.org/pdf/1512.03385)
    'epoch_num': 50,  # number of epochs
    'lr': 1e-1,  # (initial) Learning Rate
    'weight_decay': 1e-4,  # L2 Penalty
    'batch_size': 256,  # batch size (depends on hardware)
    'momentum': 0.9,

    'hierarchy': 0,  # Choose 0 for manufacturer classification, 1 for model classification
    'val_split': 10000,  # (float) Fraction of validation holdout / (int) Absolute number of data points in holdout

    'resnet': resnet_cfg[resnet_type],  # ResNet configuration

    'use_train_test_split': False  # True: use prepared split, False: use total dataset
}


def set_device():
    if torch.cuda.is_available():
        params["device"] = torch.device("cuda")  # option for NVIDIA GPUs
    elif torch.backends.mps.is_available():
        params["device"] = torch.device("mps")  # option for Mac M-series chips (GPUs)
    else:
        params["device"] = torch.device("cpu")  # default option if none of the above devices are available

    print("Device: {}".format(params["device"]))


def main():
    set_device()

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

    data_transforms = {  # TODO: TUNE
        'train': v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomChoice([
                v2.Resize(256),
                v2.Resize(224),
                v2.Resize(320)
            ]),
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(224),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            v2.ToDtype(torch.float16, scale=True),
            v2.Normalize(train_mean, train_std)
        ]),
        'val': v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float16, scale=True),
            v2.Normalize(val_mean, val_std)
        ])
    }

    wrapped_datasets = {
        'train': WrapperDataset(datasets['train'], transform=TwoCropTransform(data_transforms['train'])),
        'val': WrapperDataset(datasets['val'], transform=data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(wrapped_datasets['train'], batch_size=params['batch_size'], shuffle=True, num_workers=4),
        'val': DataLoader(wrapped_datasets['val'], batch_size=params['batch_size'], shuffle=False, num_workers=4)
    }

    print(f"Training dataset size: {len(wrapped_datasets['train'])}")
    print(f"Validation dataset size: {len(wrapped_datasets['val'])}")

    x, y = next(iter(dataloaders['train']))
    print(f"Batch of training images shape: {x[0].shape}")
    print(f"Batch of training labels shape: {y.shape}")

    x, y = next(iter(dataloaders['val']))
    print(f"Batch of validation images shape: {x.shape}")
    print(f"Batch of validation labels shape: {y.shape}")

    # Set up SupConResNet model
    sup_con_resnet = SupConResNet(len(total_set.classes), resnet_type=resnet_type).to(params['device'])

    # Loss and Optimizer
    criterion = SupConLoss(params["device"], temperature=0.1)
    optimizer = torch.optim.SGD(
        sup_con_resnet.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        momentum=params['momentum']
    )

    # LR scheduler (using Top-1-Accuracy as Validation metric)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, min_lr=1e-4, patience=2,
                                                           threshold=1e-3)

    # Save performance metrics
    train_losses = list()

    CHECKPOINT_PATH = './training_checkpoints/checkpoint.pth'
    START_FROM_CHECKPOINT = False  # set to TRUE to start from checkpoint
    start_epoch = 0

    if START_FROM_CHECKPOINT:
        checkpoint = torch.load(CHECKPOINT_PATH)
        start_epoch = checkpoint['epoch']
        sup_con_resnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_losses = checkpoint['train_losses']

    # Just some fancy progress bars # FIXME: inside epoch progress bar not working reliably for me
    pbar_epoch = trange(start_epoch, params["epoch_num"], initial=start_epoch, total=params["epoch_num"],
                        desc="Training", position=0, leave=True, unit="epoch",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]")
    pbar_inside_epoch = trange(0, (len(dataloaders['train']) + len(dataloaders['val'])),
                               desc="Training and validation per epoch", position=1, leave=False,
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]")

    # Stop the training phase in case there is no improvement
    # early_stopper = EarlyStopper(patience=10, min_delta=0.1)

    for epoch in pbar_epoch:
        pbar_inside_epoch.reset()

        # Training
        train_results = train(dataloaders['train'], sup_con_resnet, epoch, criterion, optimizer, params["device"],
                              pbar=pbar_inside_epoch, sup_con=True)
        train_losses.append(train_results[0])

        # ReduceLROnPlateau scheduler (reduce LR by 10 when top-1-accuracy does not improve)
        scheduler.step(train_results[0])
        print("\nCurrent Learning Rate: ", round(scheduler.get_last_lr()[0], 4), "\n")

        # Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': sup_con_resnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
        }, CHECKPOINT_PATH)

        # Comment on the following lines if you don't want to stop early in case of no improvement
        # if early_stopper.early_stop(validation_results[0]):
        #     params['epoch_num'] = epoch
        #     print("\n\nEarly stopping...")
        #     break

    pbar_inside_epoch.close()


if __name__ == '__main__':
    main()
