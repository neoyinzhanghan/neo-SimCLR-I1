import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from timm.optim.lars import Lars
from scheduler import CosineAnnealingLRWithWarmUp

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch SimCLR")
parser.add_argument(
    "-data", metavar="DIR", default="./datasets", help="path to dataset"
)
parser.add_argument(
    "--dataset_name",
    default="stl10",
    help="dataset name",
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=12,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--warm_up_epochs", default=10, type=int, help="number of linear warm-up epochs"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=1024,
    type=int,
    metavar="N",
    help="mini-batch size, this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--base_lr",
    default=0.3,
    type=float,
    metavar="LR",
    help="initial base learning rate, the learning rate will be scaled  based on batch size/256",
    dest="base_lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-6,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-6)",
    dest="weight_decay",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "--fp16-precision",
    action="store_true",
    help="Whether or not to use 16-bit precision GPU training.",
)

parser.add_argument(
    "--out_dim", default=1024, type=int, help="feature dimension (default: 1024)"
)
parser.add_argument(
    "--log-every-n-steps", default=10, type=int, help="Log every n steps"
)
parser.add_argument(
    "--temperature",
    default=0.07,
    type=float,
    help="softmax temperature (default: 0.07)",
)
parser.add_argument(
    "--n_views",
    default=2,
    type=int,
    metavar="N",
    help="Number of views for contrastive learning training.",
)
parser.add_argument("--gpu_index", default=0, type=int, help="Gpu index.")
parser.add_argument(
    "--num_images_per_epoch",
    default=40960,
    type=int,
    help="Number of images per epoch.",
)


def main():
    args = parser.parse_args()
    assert (
        args.n_views == 2
    ), "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device("cpu")
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset(args.data)

    train_dataset = dataset.get_dataset(
        args.dataset_name, args.n_views, args.num_images_per_epoch
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    # optimizer = torch.optim.Adam(
    #     model.parameters(), args.lr, weight_decay=args.weight_decay
    # ) # we are changing from Adam to LARS

    lr = args.base_lr * args.batch_size / 256

    optimizer = Lars(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLRWithWarmUp(
        optimizer,
        T_max=len(train_loader),
        warm_up_epochs=args.warm_up_epochs,
        eta_min=0,
        last_epoch=-1,
    )

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(
            model=model, optimizer=optimizer, scheduler=scheduler, args=args
        )
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
