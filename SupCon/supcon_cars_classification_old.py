from torchvision.transforms import v2, transforms
from tqdm import trange

# from custom files
from dataset import CompCarsImageFolder, WrapperDataset, TwoCropTransform
from moco import resnet_cfg, SupConResNet
from models import train
from utils import *
from losses import SupConLoss

# EDO'S PATHS
# root = '/Volumes/EDO/NNDL/CompCars dataset/data/image/'
# train_file = '/Volumes/EDO/NNDL/CompCars dataset/data/train_test_split/classification/train.txt'
# test_file = '/Volumes/EDO/NNDL/CompCars dataset/data/train_test_split/classification/test.txt'

# DEFAULT PATHS
root = '/home/ubuntu/data/image/'
train_file = '/home/ubuntu/data/train_test_split/classification/train.txt'
test_file = '/home/ubuntu/data/train_test_split/classification/test.txt'

resnet_type = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50'

params = {
    # Training Params (inspired from original resnet paper: https://arxiv.org/pdf/1512.03385)
    'epoch_num': 50,  # number of epochs
    'lr': 1e-1,  # (initial) Learning Rate
    'weight_decay': 1e-4,  # L2 Penalty
    'batch_size': 512,  # batch size (depends on hardware)
    'momentum': 0.9,

    'hierarchy': 0,  # Choose 0 for manufacturer classification, 1 for model classification
    'val_split': 10000,  # (float) Fraction of validation holdout / (int) Absolute number of data points in holdout

    'resnet': resnet_cfg[resnet_type],  # ResNet configuration

    'use_train_test_split': False,  # True: use prepared split, False: use total dataset

    'use_amp': True,  # Automatic Mixed Precision (AMP) for faster training on NVIDIA GPUs
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


def set_loader() -> tuple[int, dict[str, DataLoader]]:
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
    if params['use_amp']:
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
    else:
        data_transforms = { # TODO: TUNE
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

    return len(total_set.classes), dataloaders


def set_model(num_classes) -> tuple[SupConResNet, SupConLoss]:
    # Set up SupConResNet model
    sup_con_resnet = (SupConResNet(num_classes, resnet_type=resnet_type))
    
    if params['use_amp']:
        sup_con_resnet.to(params['device'], dtype=torch.float16)
    else:
        sup_con_resnet.to(params['device'])

    # Loss and Optimizer
    criterion = SupConLoss(params["device"], temperature=0.1)

    return sup_con_resnet, criterion


def set_optimizer(model) -> tuple[torch.optim, torch.optim.lr_scheduler]:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        momentum=params['momentum']
    )

    # LR scheduler (using Top-1-Accuracy as Validation metric)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, min_lr=1e-4, patience=2,
                                                           threshold=1e-3)
    return optimizer, scheduler


def save_model(model: SupConResNet, optimizer: torch.optim, epoch: int):
    MODEL_PATH = './trained_models/supcon_weights_car_'

    if params['hierarchy'] == 0:
        MODEL_PATH += 'makers_'
    else:
        MODEL_PATH += 'models_'
        
    MODEL_PATH += str(params['batch_size']) + '.pth'

    torch.save({
        'params': params,
        'model_state_dict' : model.state_dict(),
        'optimizer': optimizer,
        'epoch': epoch,
        'train_losses': train_losses
        }, MODEL_PATH)


def main():
    set_device()

    num_classes, dataloaders = set_loader()

    sup_con_resnet, criterion = set_model(num_classes)

    optimizer, scheduler = set_optimizer(sup_con_resnet)

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

    for epoch in pbar_epoch:
        pbar_inside_epoch.reset()

        # Training
        train_results = train(dataloaders['train'], sup_con_resnet, epoch, criterion, optimizer, params["device"],
                              pbar=pbar_inside_epoch)
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

    pbar_inside_epoch.close()

    save_model(sup_con_resnet, optimizer, params["epoch_num"])


if __name__ == '__main__':
    main()
