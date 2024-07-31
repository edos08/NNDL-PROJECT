from torchvision import transforms
from tqdm import trange
import torch.nn as nn

# from custom files
from supcon_dataset import CompCarsImageFolder, WrapperDataset, CIFAR10Instance
from resnet import ResNet, resnet_cfg, train, validate
from utils import *

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# DEFAULT PATHS
root = '/home/ubuntu/data/image/'
train_file = '/home/ubuntu/data/train_test_split/classification/train.txt'
test_file = '/home/ubuntu/data/train_test_split/classification/test.txt'

resnet_type = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50'

params = {
    'resnet': resnet_cfg[resnet_type],  # ResNet configuration

    # Training Params (inspired from original resnet paper: https://arxiv.org/pdf/1512.03385)
    'epoch_num': 20,                    # number of epochs
    'lr': 0.1,                          # (initial) Learning Rate
    'weight_decay': 1e-4,               # L2 Penalty
    'batch_size': 256,                  # batch size (depends on hardware)
    'momentum': 0.9,                    # SGD momentum
    'warm_up': 5,                       # number of warm-up epochs

    # MoCo Params (inspired from original MoCo paper: https://arxiv.org/abs/1911.05722)
    'moco_dim': 128,                    # feature dimension
    'moco_k': 1024,                     # queue size (carmaker: 1024, car model: 8192)
    'moco_m': 0.999,                    # momentum for updating key encoder
    'moco_t': 0.1,                      # temperature parameter (commonly 0.07 or 0.1)
    'mlp': False,                       # use mlp head

    'hierarchy': 0,                     # Choose 0 for manufacturer classification, 1 for model classification
    'val_split': 10000,                 # (float) Fraction of validation holdout / (int) Absolute number of data points in holdout
    'use_train_test_split': False,      # True: use prepared split, False: use total dataset

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

    # FOR TEST PURPOSES
    # augmentation_train = [
    #     transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                                 std=[0.2023, 0.1994, 0.2010])
    # ]

    # augmentation_val = [
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                                 std=[0.2023, 0.1994, 0.2010])
    # ]

    wrapped_datasets = {
        'train': WrapperDataset(datasets['train'], transform=data_transforms['train']),
        'val': WrapperDataset(datasets['val'], transform=data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(wrapped_datasets['train'], batch_size=params['batch_size'], shuffle=True, num_workers=os.cpu_count(), drop_last=True),
        'val': DataLoader(wrapped_datasets['val'], batch_size=params['batch_size'], shuffle=False, num_workers=os.cpu_count())
    }

    print(f"Training dataset size: {len(wrapped_datasets['train'])}")
    print(f"Validation dataset size: {len(wrapped_datasets['val'])}")

    x, y, _ = next(iter(dataloaders['train']))
    print(f"Batch of training images shape: {x[0].shape}")
    print(f"Batch of training labels shape: {y.shape}")

    x, y, _ = next(iter(dataloaders['val']))
    print(f"Batch of validation images shape: {x.shape}")
    print(f"Batch of validation labels shape: {y.shape}")

    # train_dataset = CIFAR10Instance(root="./cifar10", train=True, download=True,
    #                                          transform=transforms.Compose(augmentation_train))
    # eval_dataset = CIFAR10Instance(root="./cifar10", train=False, download=True,
    #                                         transform=transforms.Compose(augmentation_val))

    # print("CPU count: ", os.cpu_count())

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=params['batch_size'], shuffle=True,
    #     num_workers=4*os.cpu_count(), pin_memory=True, drop_last=True)
    # eval_loader = torch.utils.data.DataLoader(
    #     eval_dataset, batch_size=params['batch_size'], shuffle=False,
    #     num_workers=4*os.cpu_count(), pin_memory=True)

    return len(total_set.classes), dataloaders


def set_model(num_classes: int, path: str) -> tuple[ResNet, nn.CrossEntropyLoss]:
    # create model
    model = ResNet(params['resnet']['block'], params['resnet']['layers'], num_classes)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if os.path.isfile(path):
        print("=> loading pre-trained model '{}'".format(path))
        checkpoint = torch.load(path, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['model_state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(path))
    else:
        print("=> no checkpoint found at '{}'".format(path))

    # load the model to the device
    model.to(params['device'])

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().to(params['device'])

    return model, criterion


def set_optimizer(model) -> tuple[torch.optim.SGD, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        momentum=params['momentum']
    )

    # LR scheduler (using Loss as Validation metric)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, min_lr=1e-4, patience=3,
                                                           threshold=1e-3)
    return optimizer, scheduler


def save_model(model, train_losses, train_acc, train_top5_acc, validation_losses, validation_acc, validation_top5_acc):
    MODEL_PATH = './trained_models/' + resnet_type + '_moco_weights_car_'

    if params['hierarchy'] == 0:
        MODEL_PATH += 'makers_'
    else:
        MODEL_PATH += 'models_'
        
    if params['use_train_test_split']:
        MODEL_PATH += 'prepared_dataset_'
    else:
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

    model, criterion = set_model(num_classes, path='./trained_models/pretrained_moco_weights_car_makers_256.pth')

    optimizer, scheduler = set_optimizer(model)

    # Save performance metrics
    train_losses, validation_losses, train_acc, validation_acc, train_top5_acc, validation_top5_acc = list(), list(), list(), list(), list(), list()

    CHECKPOINT_PATH = './training_checkpoints/lin_checkpoint.pth'
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

    # Just some fancy progress bars with Telegram support for tracking training progress
    # token = "7201508620:AAFKipOQ7_Xdcgid1xDf60fCCkJuKAcPVBw"
    # chat_id = "-4239730104"

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

        # ReduceLROnPlateau scheduler (reduce LR by 10 when loss does not improve)
        scheduler.step(train_results[0])
        print("\nCurrent Learning Rate: ", round(scheduler.get_last_lr()[0], 4), "\n")

        # Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
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
