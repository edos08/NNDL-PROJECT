import torch
import os
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import transforms
from tqdm.notebook import trange

# from custom files
from dataset import CompCarsImageFolder, WrapperDataset
from models import ResNet, resnet_cfg
from models import train, validate
from utils import fix_all_seeds, train_val_dataset, compute_mean_std_from_dataset

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# DEFAULT PATHS
root = '/home/ubuntu/data/image/'

resnet_type = 'resnet18'                # 'resnet18', 'resnet34', 'resnet50'    

params = {                              ## Training Params (inspired from original resnet paper: https://arxiv.org/pdf/1512.03385)
    'epoch_num': 50,                    # number of epochs
    'lr': 1e-1,                         # (initial) Learning Rate
    'weight_decay': 1e-4,               # L2 Penalty
    'batch_size': 256,                  # batch size (tune depending on hardware)
    'momentum': 0.9,
    
    'hierarchy': 0,                     # Choose 0 for manufacturer classification, 1 for model classification
    'val_split': 10000,                 # (float) Fraction of validation holdout set / (int) Absolute number of data points in holdout set
    
    'resnet': resnet_cfg[resnet_type],  # ResNet configuration

    'seed': 28,                         # for reproducibility (NOTE: may be still non-deterministic for multithreaded/multiprocess stuff, e.g. DataLoader)
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
    datasets = train_val_dataset(total_set, val_split=params['val_split'], seed=params['seed'])

    '''
    NOTE: This computation takes some time. Could be accelerated by:
    1. Using dataloader (vectorized batches)
    2. Resize images before computing statistics
    3. Use fixed values computed once (with fixed seed) TODO 
    '''
    print("Computing normalization statistics... (this might take a while)")

    # default (computed statistic on whole dataset)
    # mean, std = [0.483, 0.471, 0.463], [0.297, 0.296, 0.302]

    # Compute mean and std for training dataset
    train_mean, train_std = compute_mean_std_from_dataset(datasets['train'])
    print(f"Training dataset mean: {train_mean}")
    print(f"Training dataset std: {train_std}")
    # train_mean, train_std = mean, std

    # Compute mean and std for validation dataset
    val_mean, val_std = compute_mean_std_from_dataset(datasets['val'])
    print(f"Validation dataset mean: {val_mean}")
    print(f"Validation dataset std: {val_std}")
    # val_mean, val_std = mean, std

    print("SAVE NEW DEFAULT NORMALIZATION VALUES FOR VALIDATION AN TRAINING DATA ONCE AND FOR ALL")

    # Augmentation Strategy similar to original ResNet paper: https://arxiv.org/pdf/1512.03385
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomChoice([       # Scale Augmentation
                transforms.Resize(256),
                transforms.Resize(224),
                transforms.Resize(320)
            ]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),     # ResNet expects 224x224 images
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),     # Evaluate using 224x224 central part of image
            transforms.ToTensor(),
            transforms.Normalize(val_mean, val_std)
        ])
    }

    wrapped_datasets = {
        'train': WrapperDataset(datasets['train'], transform=data_transforms['train']),
        'val': WrapperDataset(datasets['val'], transform=data_transforms['val'])
    }

    # NOTE: for num_workers != 0 dataloader won't be deterministic (for reproducible implementation see https://pytorch.org/docs/stable/notes/randomness.html)
    dataloaders = {
        'train': DataLoader(wrapped_datasets['train'], batch_size=params['batch_size'], shuffle=True, num_workers=os.cpu_count()), # NOTE: num_workers to tune for performance or reproducability
        'val': DataLoader(wrapped_datasets['val'], batch_size=params['batch_size'], shuffle=False, num_workers=os.cpu_count())
    }

    print(f"Training dataset size: {len(wrapped_datasets['train'])}")
    print(f"Validation dataset size: {len(wrapped_datasets['val'])}")

    x, y = next(iter(dataloaders['train']))
    print(f"Batch of training images shape: {x.shape}")
    print(f"Batch of training labels shape: {y.shape}")

    x, y = next(iter(dataloaders['val']))
    print(f"Batch of validation images shape: {x.shape}")
    print(f"Batch of validation labels shape: {y.shape}")
    
    return len(total_set.classes), dataloaders


def set_model(num_classes: int) -> tuple[ResNet, torch.nn.CrossEntropyLoss]:

    model = ResNet(params['resnet']['block'], params['resnet']['layers'], 
            num_classes).to(params['device'])
    
    criterion = torch.nn.CrossEntropyLoss().to(params['device'])

    return model, criterion


def set_optimizer(model: ResNet) -> tuple[torch.optim.SGD, torch.optim.lr_scheduler.ReduceLROnPlateau]:

    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=params['lr'], 
        weight_decay=params['weight_decay'], 
        momentum=params['momentum']
    )

    # optimizer = torch.optim.Adam(
    #     resnet.parameters(), 
    #     lr=params['lr'], 
    #     weight_decay=params['weight_decay']
    # )

    # LR scheduler (using Top-1-Accuracy as Validation metric)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, min_lr=1e-4, patience=3, threshold=1e-3)

    return optimizer, scheduler


def save_model(model: ResNet, train_losses: list, train_acc: list, train_top5_acc: list, validation_losses: list, validation_acc: list, validation_top5_acc: list):
    MODEL_PATH = './trained_models/' + resnet_type + '_weights_car_'

    if params['hierarchy'] == 0:
        MODEL_PATH += 'makers_'
    else:
        MODEL_PATH += 'models_'
        
    MODEL_PATH += 'full_dataset_'
        
    MODEL_PATH += str(params['batch_size']) + '.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_acc': train_acc,
        'train_top5_acc': train_top5_acc,
        'validation_losses': validation_losses,
        'validation_acc': validation_acc,
        'validation_top5_acc': validation_top5_acc,
        'resnet': params['resnet'],
        }, MODEL_PATH)


def main():
    # !!! NOTE: REMEMBER TO PASS SEED TO train_val_dataset FUNCTION AS ARGUMENT !!! 
    fix_all_seeds(seed=params['seed'])

    set_device()

    num_classes, dataloaders = set_loader()

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    model, criterion = set_model(num_classes)

    optimizer, scheduler = set_optimizer(model)

    # Save performance metrics
    train_losses, validation_losses, train_acc, validation_acc, train_top5_acc, validation_top5_acc = list(), list(), list(), list(), list(), list()


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
        train_acc = checkpoint['train_acc']
        train_top5_acc = checkpoint['train_top5_acc']
        validation_losses = checkpoint['validation_losses']
        validation_acc = checkpoint['validation_acc']
        validation_top5_acc = checkpoint['validation_top5_acc']

    # Just some fancy progress bars
    pbar_epoch = trange(start_epoch, params["epoch_num"], initial=start_epoch, total=params["epoch_num"], desc="Training",
                        unit="epoch", position=0, leave=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]")
    pbar_inside_epoch = trange(0, (len(train_loader)+len(val_loader)), desc="Training and validation per epoch", position=1, leave=False,
                               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]")

    for epoch in pbar_epoch:
        pbar_inside_epoch.reset()

        # Training
        train_results = train(train_loader, model, epoch, criterion, optimizer, params["device"], pbar=pbar_inside_epoch)
        train_losses.append(train_results[0])
        train_acc.append(1 - train_results[1])                 # saving acc error
        train_top5_acc.append(1 - train_results[2])

        # Validation
        validation_results = validate(val_loader, model, epoch, criterion, params["device"], pbar=pbar_inside_epoch)
        validation_losses.append(validation_results[0])
        validation_acc.append(1 - validation_results[1])       # saving acc error
        validation_top5_acc.append(1 - validation_results[2])

        # Scheduler
        scheduler.step(validation_results[1])   # ReduceLROnPlateau scheduler (reduce LR by 10 when top-1-accuracy does not improve)
        print("\nCurrent Learning Rate: ", round(scheduler.get_last_lr()[0], 4), "\n")

        # Checkpoint
        torch.save({
            'epoch' : epoch + 1,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'train_losses': train_losses,
            'train_acc': train_acc,
            'train_top5_acc': train_top5_acc,
            'validation_losses': validation_losses,
            'validation_acc': validation_acc,
            'validation_top5_acc': validation_top5_acc,
        }, CHECKPOINT_PATH)

    pbar_inside_epoch.close()

    save_model(model, train_losses, train_acc, train_top5_acc, validation_losses, validation_acc, validation_top5_acc)


if __name__ == '__main__':
    main()


