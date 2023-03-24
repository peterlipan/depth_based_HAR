import os
import wandb
import argparse
import torch.optim
from models import TSN
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from data import TSNDataSet, Transforms
from utils import yaml_config_hook, train
import numpy as np
from torch.nn import DataParallel


def main(args, wandb_logger):
    data_length = None
    if args.modality == 'depth':
        data_length = 1
    elif args.modality in ['Flow', 'depthDiff', 'gradientSum']:
        data_length = 5

    # define dataset
    transforms = Transforms(args.modality, args.img_size)
    train_dataset = TSNDataSet(data_path=args.data_path, csv_path=args.csv_file_train,
                               num_segments=args.num_segments, new_length=data_length,
                               modality=args.modality, image_tmpl=args.image_tmpl,
                               transform=transforms.train_transforms, test_mode=False)

    test_dataset = TSNDataSet(data_path=args.data_path, csv_path=args.csv_file_test,
                              num_segments=args.num_segments, new_length=data_length,
                              modality=args.modality, image_tmpl=args.image_tmpl,
                              transform=transforms.test_transforms, test_mode=True)
    num_class = train_dataset.num_class

    # init model
    model = TSN(num_class, args.num_segments, args.modality, new_length=data_length,
                backbone=args.backbone, dropout=args.dropout, partial_bn=args.partialbn)
    policies = model.get_optim_policies()
    model = DataParallel(model, device_ids=[int(x) for x in args.gpus.split(',')]).cuda()
    cudnn.benchmark = True

    # Dataloaders
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    dataloaders = (train_loader, test_loader)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train(loaders=dataloaders, model=model, criterion=criterion, optimizer=optimizer,
          logger=wandb_logger, args=args)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="Ours")
    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    parser.add_argument('--config_file', type=str, help='path to the yaml file')
    temp = parser.parse_args()

    yaml_config = yaml_config_hook(temp.config_file)
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb
    if not args.debug:
        wandb.login(key="cb1e7d54d21d9080b46d2b1ae2a13d895770aa29")
        config = dict()

        for k, v in yaml_config.items():
            config[k] = v

        logger = wandb.init(
            project="%s_%s_%s" % (args.dataset, args.modality, args.task),
            notes="Sensys 2023 depth HAR",
            tags=["baseline", "Sensys2023", "depth camera", "human action recognition"],
            config=config
        )
    else:
        logger = None

    main(args, logger)
