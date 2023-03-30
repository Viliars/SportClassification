import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from pathlib import Path


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_transform(size: int):
    pipeline = []
    pipeline.append(transforms.Resize(size, interpolation=3))
    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(pipeline)


class SportDataset(Dataset):
    '''
    # -----------------------------------------
    # dataset for Sport Classification
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(SportDataset, self).__init__()
        self.opt = opt

        self.csv_path = opt["csv_path"]
        self.dataroot = Path(opt["dataroot"])

        self.data = pd.read_csv(self.csv_path)

        self.classes = sorted(list(set(self.data["label"])))
        self.class2id = {C: i for i, C in enumerate(self.classes)}
        self.id2class = {i: C for i, C in enumerate(self.classes)}
    
        self.images = list(self.data["image_id"])
        self.labels = list(self.data["label"])

        self.transform = build_transform(opt["size"])

    def __getitem__(self, index):
        image_path = self.dataroot / self.images[index]
        class_id = self.class2id[self.labels[index]]

        image = default_loader(image_path)

        transformed_image = self.transform(image)

        return transformed_image, class_id

    def __len__(self):
        return len(self.images)
    
