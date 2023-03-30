import click
import random
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import numpy as np
import math
from utils import utils_option as option
from loguru import logger
from data.select_dataset import define_Dataset
from models.select_model import define_Model


@click.command()
@click.option('--opt', default='options/nextvit_small.yml', help='Path to option YAML file.')
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

    for phase, dataset_opt in opt['datasets'].items():
        print(phase, dataset_opt)
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))

            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))

            train_loader = DataLoader(train_set,
                                        batch_size=dataset_opt['dataloader_batch_size'],
                                        shuffle=dataset_opt['dataloader_shuffle'],
                                        num_workers=dataset_opt['dataloader_num_workers'],
                                        drop_last=True,
                                        pin_memory=True)

        elif phase == 'val':
            val_set = define_Dataset(dataset_opt)
            val_size = int(math.ceil(len(val_set) / dataset_opt['dataloader_batch_size']))
            val_loader = DataLoader(val_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=dataset_opt['dataloader_shuffle'], num_workers=dataset_opt['dataloader_num_workers'],
                                     drop_last=False, pin_memory=True)
            
            logger.info('Number of val images: {:,d}, iters: {:,d}'.format(len(val_set), val_size))
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)
        
    model = define_Model(opt)
    class_weight = torch.Tensor(opt["train"]["weight"])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction="mean")

    # for epoch in range(1000000):  # keep running
    #     model.train()
    #     for i, train_data in enumerate(train_loader):
    #         images, labels = train_data

    #         images = images.to(device, non_blocking=True)
    #         labels = labels.to(device, non_blocking=True)

    #         with autocast(enabled=True):
    #             outputs = model(images)
    #             loss = criterion(outputs, labels)

    #         loss_value = loss.item()

    #         optimizer.zero_grad()

    #         # TODO do train pipeline




if __name__ == '__main__':
    main()
