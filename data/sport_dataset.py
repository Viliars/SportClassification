import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from pathlib import Path
from timm.data import create_transform


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_train_transform(opt):
    pipeline = create_transform(
        input_size=opt["input_size"],
        is_training=True,
        color_jitter=opt["color_jitter"],
        auto_augment=opt["auto_augment"],
        interpolation="bicubic",
        re_prob=opt["reprob"],
        re_mode=opt["remode"],
        re_count=opt["recount"],
    )

    return pipeline


def build_val_transform(opt):
    pipeline = []

    pipeline.append(transforms.Resize(256, interpolation=3))
    pipeline.append(transforms.CenterCrop(opt["input_size"]))

    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(pipeline)


class SportDataset(Dataset):
    """
    # -----------------------------------------
    # dataset for Sport Classification
    # -----------------------------------------
    """

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

        self.ids = [self.class2id[C] for C in self.labels]

        if opt["name"] == "train_dataset":
            self.transform = build_train_transform(self.opt["augs"])
        else:
            self.transform = build_val_transform(self.opt["augs"])

    def __getitem__(self, index):
        image_path = self.dataroot / self.images[index]
        class_id = self.class2id[self.labels[index]]

        image = default_loader(image_path)

        transformed_image = self.transform(image)

        return transformed_image, class_id

    def __len__(self):
        return len(self.images)
