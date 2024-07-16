from PIL import Image
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional
from torchvision.datasets import DatasetFolder

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class CompCarsImageFolder(DatasetFolder):
# Same as ImageFolder class, except for overriden find_classes method.
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        hierarchy:int = 0,
    ):
        self.hierarchy = hierarchy  # needs to be initialized BEFORE calling super
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )
        self.imgs = self.samples


    def find_classes(self, directory: str | Path) -> Tuple[List[str], Dict[str, int]]:          
        if self.hierarchy == 1: # descend one folder hierarchy to create classes
            classes = []
            for dir in os.scandir(directory):
                if dir.is_dir():
                    classes.extend(dir.name + os.sep + entry.name for entry in os.scandir(os.path.join(directory, dir.name)) if entry.is_dir())       
            classes.sort()
       
        elif self.hierarchy == 2: # descend two folder hierarchies
            classes = []
            for supDir in os.scandir(directory):
                if supDir.is_dir():
                    for dir in os.scandir(os.path.join(directory, supDir.name)):
                        if dir.is_dir():
                            classes.extend(supDir.name + os.sep + dir.name + os.sep + entry.name for entry in os.scandir(os.path.join(directory, supDir.name, dir.name)) if entry.is_dir())
            classes.sort()

        # fallthrough to default implementation
        else: # use directories in root as classes
            classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx
    
class WrapperDataset:
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.dataset)