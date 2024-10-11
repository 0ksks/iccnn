import random
from collections import OrderedDict

import typer
import torch
import os
import wandb
from warnings import filterwarnings
from torchvision.models import vgg16_bn
from torch.utils.data import Subset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn.cluster import SpectralClustering

from data.CUB_200_2011 import get_dataset as get_cub_dataset, single_category_dataset
from data.voc2010_crop import get_dataset as get_voc_dataset
from data import get_dataloader
from network.VGG16BN import VGG16BN
from global_variable import get_config_value, parse_config_path, RUN_NAME

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "2"

filterwarnings("ignore")


def main(
        subset_size: int,
        batch_size: int,
        num_workers: int,
        center_num: int,
        cluster_interval: int,
        cluster_loss_factor: float,
        cluster_stop_epoch: int,
        hierarchical_loss_factor: float,
        log_tmp_output_every_step: int,
        log_tmp_output_every_epoch: int,
        save_pth: int,
        save_pth_path: str,
        save_pth_name: str,
        save_feature_map: int,
        max_epochs: int,
        wandb_online: int,
        run_name: str,
):
    num_classes = 2
    weight = "vgg_16_bn"
    dataset = "voc_2010_crop"

    vgg16 = vgg16_bn(weights=None)
    state_dict = torch.load(
        parse_config_path(
            get_config_value("weight.root") +
            get_config_value(f"weight.{weight}")
        ),
        weights_only=True,
        map_location=torch.device("cpu"),  # TODO gpu
    )

    if weight == "model_2499":  # TODO why change the original fc layers?
        vgg16.classifier[3] = torch.nn.Linear(
            in_features=4096,
            out_features=512
        )
        vgg16.classifier[6] = torch.nn.Linear(  # change to target class number
            in_features=vgg16.classifier[3].out_features,
            out_features=num_classes
        )
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.module.', '')
            new_state_dict[name] = v
        state_dict = new_state_dict

    vgg16.load_state_dict(state_dict)

    if wandb_online == 0:
        os.environ["WANDB_MODE"] = "offline"
    wandb.login(key=get_config_value("wandb.api_key"))
    wandb_log_dir = parse_config_path(get_config_value("wandb.root"))
    if not os.path.exists(wandb_log_dir):
        os.makedirs(wandb_log_dir)
    wandb.init(
        name=run_name,
        dir=wandb_log_dir
    )
    wandb_logger = WandbLogger(
        project=get_config_value("wandb.project"),
        name=run_name,
        save_dir=wandb_log_dir
    )

    dataset_path = parse_config_path(
        get_config_value("dataset.root") +
        get_config_value(f"dataset.{dataset}")
    )

    if dataset == "cub_200_2011":
        datasets = get_cub_dataset(
            dataset_path
        )
        for idx, dataset in enumerate(datasets):
            datasets[idx] = single_category_dataset(dataset)
    elif dataset == "voc_2010_crop":
        datasets = get_voc_dataset(dataset_path, "bird")
    else:
        raise Exception("dataset not supported")

    train_dataset = [datasets[0], ]

    if subset_size:
        train_dataset[0] = Subset(
            train_dataset[0],
            random.sample(
                range(len(train_dataset[0])),
                subset_size
            )
        )

    if not num_workers:
        num_workers = os.cpu_count() // 2

    if not log_tmp_output_every_step:
        log_tmp_output_every_step = None
    if not log_tmp_output_every_epoch:
        log_tmp_output_every_epoch = None

    example_input, _ = datasets[1][0]
    if cluster_stop_epoch == 0:
        cluster_stop_epoch = max_epochs

    if save_pth == "0":
        save_pth = None
    if save_pth_path == "0":
        save_pth_path = None

    if save_pth and not save_pth_path:
        save_pth_path = parse_config_path(get_config_value("weight.root"))

    model = VGG16BN(
        model=vgg16, center_num=center_num, num_classes=num_classes,
        example_input=example_input.unsqueeze(0), train_dataloader=get_dataloader(
            train_dataset,
            train_kwargs=dict(
                num_workers=num_workers,
                persistent_workers=True
            ),
            split="train",
            batch_size=batch_size
        ),
        cluster_class=SpectralClustering, cluster_interval=cluster_interval,
        cluster_loss_factor=cluster_loss_factor, cluster_stop_epoch=cluster_stop_epoch,
        hierarchical_loss_factor=hierarchical_loss_factor,
        log_tmp_output_every_step=log_tmp_output_every_step,
        log_tmp_output_every_epoch=log_tmp_output_every_epoch,
        save_pth=save_pth,
        save_pth_path=save_pth_path,
        save_pth_name=save_pth_name,
        overwrite_classifier=weight != "tuned",
        save_feature_map=save_feature_map
    )

    wandb.log(
        {
            "cluster_loss_factor": cluster_loss_factor,
            "hierarchical_loss_factor": hierarchical_loss_factor,
        },
        step=0
    )

    trainer = Trainer(logger=wandb_logger, max_epochs=max_epochs, accelerator="auto")
    trainer.fit(model)

    wandb.teardown()


def cli(
        subset_size: int = typer.Option(
            32,
            "--subset-size", "-ss",
            help="the subset size of the training set, set `0` to use all",
            prompt="subset size(`0` to use all)"
        ),
        batch_size: int = typer.Option(
            4,
            "--batch-size", "-bs",
            help="the batch size for training",
            prompt="batch size"
        ),
        num_workers: int = typer.Option(
            0,
            "--num-workers", "-nw",
            help="the number of workers for training, set `0` to use half",
            prompt="num workers(`0` to use half)"
        ),
        center_num: int = typer.Option(
            5,
            "--center-num", "-cn",
            help="the number of cluster center when clustering",
            prompt="center num"
        ),
        cluster_interval: int = typer.Option(
            1,
            "--cluster-loss-interval", "-cli",
            help="the interval between clustering loss updates",
            prompt="cluster loss interval"
        ),
        cluster_loss_factor: float = typer.Option(
            1e-1,
            "--cluster-loss-factor", "-clf",
            help="the loss factor on clustering loss when adding up all losses",
            prompt="cluster loss factor"
        ),
        cluster_stop_epoch: int = typer.Option(
            200,
            "--cluster-stop-epoch", "-cse",
            help="stop clustering after this epoch",
            prompt="cluster stop epoch"
        ),
        hierarchical_loss_factor: float = typer.Option(
            0.1,
            "--hierarchical-loss-factor", "-hlf",
            help="the loss factor on hierarchical clustering loss when adding up all losses",
            prompt="hierarchical loss factor"
        ),
        log_tmp_output_every_step: int = typer.Option(
            0,
            "--log-tmp-output-every-step", "-step",
            help="to log tmp output every step, set `0` to disable log",
            prompt="log tmp output every step(`0` to disable)"
        ),
        log_tmp_output_every_epoch: int = typer.Option(
            0,
            "--log-tmp-output-every-epoch", "-epoch",
            help="to log tmp output every epoch, set `0` to disable log",
            prompt="log tmp output every epoch(`0` to disable)"
        ),
        save_pth: int = typer.Option(
            0,
            "--save-pth", "-sw",
            help="to save model weights, set `0` to disable",
            prompt="save model weights or not(`0` to disable)"
        ),
        save_pth_path: str = typer.Option(
            "0",
            "--save-pth-path", "-swp",
            help="where to save model weights, set `0` to use default dir",
            prompt=f"save path(`0` to default `{parse_config_path(get_config_value('weight.root'))}`)"
        ),
        save_pth_name: str = typer.Option(
            "0",
            "--save-pth-name", "-swn",

            help="name of the model weights, set `0` to use default name",
            prompt="save name(`0` to default `model_${global_step}.pth`)"
        ),
        save_feature_map: int = typer.Option(
            0,
            "--save-feature-map", "-epoch",
            help="to log feature map every epoch, set `0` to disable log",
            prompt="log feature map every epoch(`0` to disable)"
        ),
        max_epochs: int = typer.Option(
            100,
            "--max-epoch", "-max",
            help="the maximum epochs for training",
            prompt="max epochs"
        ),
        wandb_online: int = typer.Option(
            0,
            "--wandb-online", "-sync",
            help="whether upload to wandb",
            prompt="wandb online"
        ),
        run_name: str = typer.Option(
            RUN_NAME,
            "--run-name", "-rn",
            help="the name of the run, default is datetime now",
            prompt="run name"
        )
):
    main(
        subset_size, batch_size, num_workers,
        center_num, cluster_interval, cluster_loss_factor, cluster_stop_epoch,
        hierarchical_loss_factor,
        log_tmp_output_every_step, log_tmp_output_every_epoch,
        save_pth, save_pth_path, save_pth_name, save_feature_map,
        max_epochs, wandb_online, run_name
    )


if __name__ == '__main__':
    # typer.run(cli)
    main(
        subset_size=8,
        batch_size=4,
        num_workers=1,
        center_num=5,
        cluster_interval=1,
        cluster_loss_factor=1e-1,
        cluster_stop_epoch=0,
        hierarchical_loss_factor=1e-3,
        log_tmp_output_every_step=0,
        log_tmp_output_every_epoch=0,
        save_feature_map=0,
        save_pth=1,
        save_pth_path="",
        save_pth_name="",
        max_epochs=2,
        wandb_online=0,
        run_name=RUN_NAME
    )
