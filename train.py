import click
import random
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import f1_score
import math
from utils import utils_option as option
from loguru import logger
from data.select_dataset import define_Dataset
from models.select_model import define_Model
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy


@click.command()
@click.option(
    "--opt", default="options/nextvit_small.yml", help="Path to option YAML file."
)
def main(opt):
    opt = option.parse(opt, is_train=True)

    device = torch.device("cuda")
    seed = opt["seed"]

    cudnn.benchmark = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    class_weights = opt["train"]["weight"]

    for phase, dataset_opt in opt["datasets"].items():
        print(phase, dataset_opt)
        if phase == "train":
            train_set = define_Dataset(dataset_opt)
            train_size = int(
                math.ceil(len(train_set) / dataset_opt["dataloader_batch_size"])
            )

            probs = [class_weights[ID] for ID in train_set.ids]

            logger.info(
                "Number of train images: {:,d}, iters: {:,d}".format(
                    len(train_set), train_size
                )
            )

            sampler = WeightedRandomSampler(probs, num_samples=len(train_set), replacement=True)

            train_loader = DataLoader(
                train_set,
                batch_size=dataset_opt["dataloader_batch_size"],
                #shuffle=dataset_opt["dataloader_shuffle"],
                num_workers=dataset_opt["dataloader_num_workers"],
                drop_last=True,
                pin_memory=True,
                sampler=sampler
            )

        elif phase == "val":
            val_set = define_Dataset(dataset_opt)
            val_size = int(
                math.ceil(len(val_set) / dataset_opt["dataloader_batch_size"])
            )
            val_loader = DataLoader(
                val_set,
                batch_size=dataset_opt["dataloader_batch_size"],
                shuffle=dataset_opt["dataloader_shuffle"],
                num_workers=dataset_opt["dataloader_num_workers"],
                drop_last=False,
                pin_memory=True,
            )

            logger.info(
                "Number of val images: {:,d}, iters: {:,d}".format(
                    len(val_set), val_size
                )
            )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    model = define_Model(opt)

    checkpoint = torch.load("experiments/large_image_net_augs_mixup_clean_all/best.pth") #TODO move path in config
    model.load_state_dict(checkpoint)

    model.to(device)

    #TODO Make loss by config
    criterion = SoftTargetCrossEntropy() #torch.nn.CrossEntropyLoss(weight=class_weight, reduction="mean")
    scaler = GradScaler()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt["train"]["lr"],
        weight_decay=opt["train"]["weight_decay"],
    )

    scheduler = lr_scheduler.MultiStepLR(optimizer, 
        opt["train"]['scheduler_milestones'], opt["train"]['scheduler_gamma']
    )

    mixup_opt = opt["datasets"]["train"]["mixup"]
    mixup = Mixup(
        mixup_alpha=mixup_opt["mixup_alpha"], cutmix_alpha=mixup_opt["cutmix_alpha"],
        cutmix_minmax=None, prob=mixup_opt["prob"], switch_prob=mixup_opt["switch_prob"],
        mode=mixup_opt["mode"], label_smoothing=0.1, num_classes=mixup_opt["num_classes"]
    )

    best_f1 = 0.0

    for epoch in range(1000000):
        model.train()
        losses = []
        for i, train_data in enumerate(train_loader):
            images, labels = train_data

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            images, labels = mixup(images, labels)

            with autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)

            losses.append(loss.item())

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

        mean_loss = np.mean(losses)

        print(f"epoch: {epoch}, Loss: {mean_loss}")

        scheduler.step()

        if epoch % opt["train"]["checkpoint_test"] == 0:
            model.eval()
            with torch.no_grad():
                all_outputs = []
                all_labels = []
                for i, val_data in enumerate(val_loader):
                    images, labels = val_data

                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with autocast(enabled=True):
                        outputs = model(images)
                    
                    all_outputs.append(outputs.argmax(dim=1).cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

                full_predicted = np.concatenate(all_outputs)
                full_labels = np.concatenate(all_labels)

                f1 = f1_score(full_labels, full_predicted, average="micro")

                print(f"epoch: {epoch}, F1: {f1}")

                if f1 > best_f1:
                    exp_dir = opt["train"]["exp_dir"]
                    torch.save(model.state_dict(), f"experiments/{exp_dir}/best.pth")
                    print(f"----- New Best model with F1 = {f1} -----")
                    best_f1 = f1

        if epoch % opt["train"]["checkpoint_save"] == 0:
            exp_dir = opt["train"]["exp_dir"]
            torch.save(model.state_dict(), f"experiments/{exp_dir}/{epoch:05d}.pth")


if __name__ == "__main__":
    main()
