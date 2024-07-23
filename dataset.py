from PIL import Image
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional
from torchvision.datasets import DatasetFolder, VisionDataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def read_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


class ImagesFromTextFile(VisionDataset):

    def __init__(self,
                 root: str | Path = None,
                 txt_file: str | Path = None,
                 transform: Callable[..., Any] | None = None,
                 target_transform: Callable[..., Any] | None = None,
                 loader: Callable[[str], Any] = pil_loader,
                 hierarchy: int = 0,
                 ) -> None:
        self.hierarchy = hierarchy
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform)

        classes, class_to_idx = self.find_classes(txt_file)  # find classes and map them to indices
        samples = self.make_dataset(self.root, txt_file,
                                    class_to_idx)  # read samples from textfile with correct class index as target

        self.loader = loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def find_classes(self, directory: str | Path) -> Tuple[List[str], Dict[str, int]]:
        classes = []

        lines = read_file(directory)

        for line in lines:
            line_parts = line.split('/')

            if self.hierarchy == 0:
                the_class = line_parts[0]

                if the_class in classes:
                    continue
                else:
                    classes.append(the_class)
                classes.sort()

            elif self.hierarchy == 1:
                the_class = line_parts[0] + os.sep + line_parts[1]
                if the_class in classes:
                    continue
                else:
                    classes.append(the_class)
                classes.sort()

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

    def make_dataset(
            self,
            root: str | Path,
            txt_file: str | Path,
            class_to_idx: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        instances = []

        lines = read_file(txt_file)

        for line in sorted(lines):
            line_parts = line.split('/')
            target_class = ""

            if self.hierarchy == 0:
                target_class = line_parts[0]
            elif self.hierarchy == 1:
                target_class = line_parts[0] + os.sep + line_parts[1]

            class_index = class_to_idx[target_class]
            rel_path = os.path.join(*line_parts)
            path = os.path.join(root, rel_path)

            item = path, class_index
            instances.append(item)

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


class CompCarsImageFolder(DatasetFolder):
    # Same as ImageFolder class, except for overridden find_classes method.

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            allow_empty: bool = False,
            hierarchy: int = 0,
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
        if self.hierarchy == 1:  # descend one folder hierarchy to create classes
            classes = []
            for dir in os.scandir(directory):
                if dir.is_dir():
                    classes.extend(
                        dir.name + os.sep + entry.name for entry in os.scandir(os.path.join(directory, dir.name)) if
                        entry.is_dir())
            classes.sort()

        # NOTE: hierarchy == 2 fails due to the fact that some years
        # are not well-defined (e.g. empty) in CompCars dataset
        # elif self.hierarchy == 2:  # descend two folder hierarchies
        #     classes = []
        #     for supDir in os.scandir(directory):
        #         if supDir.is_dir():
        #             for dir in os.scandir(os.path.join(directory, supDir.name)):
        #                 if dir.is_dir():
        #                     classes.extend(supDir.name + os.sep + dir.name + os.sep + entry.name
        #                     for entry in os.scandir(os.path.join(directory, supDir.name, dir.name)) if entry.is_dir())
        #     classes.sort()

        # fallthrough to default implementation
        else:  # use directories in root as classes
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
