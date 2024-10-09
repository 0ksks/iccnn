import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch import nn

from network.component.LitNetwork import LitModel
from network.component.CustomPad2d import overwrite_conv_2d
from network.component.SMGBlock import SMGBlock

from network.util.similarity import Similarity
from network.util.clustering import Clustering
from network.util.interpolation import interpolate_grid

from network.log.contribution_norm import *
from network.log.feature_map import *

from pretty_print import pretty_print
from global_variable import EPSILON


class VGG16BN(LitModel):
    def __init__(
            self,
            model,
            center_num: int,
            num_classes: int,
            example_input: torch.Tensor,
            train_dataloader: torch.utils.data.DataLoader,
            cluster_class,
            cluster_loss_interval=1,
            cluster_loss_factor=0.1,
            cluster_stop_epoch=200,
            log_tmp_output_every_step=None,
            log_tmp_output_every_epoch=None,
            overwrite_classifier=True,
            save_feature_map=False
    ):
        model = overwrite_conv_2d(model)  # use custom pad 2d
        if overwrite_classifier:
            model.classifier[6] = torch.nn.Linear(  # change to target class number
                in_features=model.classifier[3].out_features,
                out_features=num_classes
            )
        super(VGG16BN, self).__init__(
            model, cluster_loss_interval, log_tmp_output_every_step,
            log_tmp_output_every_epoch, example_input
        )

        #  config cluster
        self.SMG_block = None
        self.center_num = center_num
        self.clustering = Clustering(self.device)
        self.clustering.set_cluster_class(cluster_class)
        self.cluster_loss_factor = cluster_loss_factor
        self.cluster_stop_epoch = cluster_stop_epoch

        # config log
        self._train_dataloader = train_dataloader
        self.add_intercept_output("features.39", "5.2")
        self.add_intercept_output("features.42", "5.3")
        self.total_filter_feature_map = None
        self.total_filter_layer_name = "features.40"
        self.save_feature_map = save_feature_map

        self.rough_cluster_mapping = None
        self.rough_intra_similarity_mask = None
        self.rough_inter_similarity_mask = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._train_dataloader

    def masked_similarity(self, correlation: torch.Tensor, mask: torch.Tensor):
        correlation, mask = correlation.to(self.device), mask.to(self.device)
        center, batch, channel, _ = correlation.shape
        similarity = torch.sum(
            correlation * mask.view(
                center, 1, channel, channel
            ).repeat(
                1, batch, 1, 1
            ),
            dim=(1, 2, 3)
        )
        return similarity

    def hierarchical_loss_fn(self, precise_f_map: torch.Tensor, rough_f_map: torch.Tensor):
        if self.rough_cluster_mapping is None:  # skip when rough_cluster_mapping hasn't been calculated
            return None

        b, c, h, w = rough_f_map.shape
        avg_rough_f_map_mask = self.rough_cluster_mapping.unique(dim=1)  # (C, cluster)
        avg_rough_f_map_mask /= avg_rough_f_map_mask.sum(dim=0)
        avg_rough_f_map_mask = (
            avg_rough_f_map_mask
            .view(1, c, self.center_num, 1, 1)
            .expand(b, c, self.center_num, h, w)
        )

        weighted_rough_f_map = (
                rough_f_map
                .unsqueeze(2)
                .expand(b, c, self.center_num, h, w)
                * avg_rough_f_map_mask
        )
        avg_rough_f_map = weighted_rough_f_map.sum(dim=1)

        if rough_f_map.shape != precise_f_map.shape:  # scaling
            avg_rough_f_map = avg_rough_f_map.numpy(force=True)
            avg_rough_f_map = interpolate_grid(avg_rough_f_map, precise_f_map.shape)
            avg_rough_f_map = torch.from_numpy(avg_rough_f_map).to(self.device)

    def cluster_loss_fn(self, feature_map: torch.Tensor, labels: torch.Tensor):
        pretty_print("calculating clustering loss... (network.VGG16BN 68)")
        batch, channel, height, width = feature_map.shape

        if not self.SMG_block:
            self.SMG_block = SMGBlock(
                first_decay=0.0,
                decay=0.95,
                channel=channel,
                feature_map_pixel=height * width,
                device=self.device
            )

        with torch.no_grad():
            pearson_correlation = self.SMG_block(feature_map)  # (B, C, C)
        pearson_similarity = (pearson_correlation + 1) / 2

        labels_mask = (1 - labels.to(torch.long)).view(batch, 1, 1)  # (B, 1, 1)

        pearson_similarity = (
                (pearson_similarity * labels_mask) / batch
        ).view(
            1, batch, channel, channel
        ).repeat(
            self.center_num, 1, 1, 1
        )  # (center_num, B, C, C)

        intra_similarity = self.masked_similarity(
            pearson_similarity, self.rough_intra_similarity_mask
        )  # (center_num)

        inter_similarity = self.masked_similarity(
            pearson_similarity, self.rough_inter_similarity_mask
        )  # (center_num)

        total_intra_similarity = torch.sum(intra_similarity)
        total_inter_similarity = torch.sum(inter_similarity)

        if total_intra_similarity == total_inter_similarity and total_inter_similarity == 0:
            cluster_loss = -1
        else:
            cluster_loss = -torch.sum(intra_similarity / (inter_similarity + EPSILON))

        return cluster_loss, torch.sum(intra_similarity) / batch, torch.sum(inter_similarity) / batch

    def training_step_loss_fn(
            self, inputs: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        classification_loss = F.cross_entropy(outputs, labels)
        cluster_loss = 0
        loss_log = {}
        if self.current_epoch % self.cluster_loss_interval == 0:
            cluster_loss, intra_similarity, inter_similarity = (
                self.cluster_loss_fn(
                    self.intercept_output["5.3"],
                    labels
                )
            )
            loss_log.update({
                "cluster_loss": cluster_loss,
                "intra_similarity": intra_similarity,
                "inter_similarity": inter_similarity,
            })
            self.hierarchical_loss_fn(self.intercept_output["5.2"], self.intercept_output["5.3"])

        total_loss = self.cluster_loss_factor * cluster_loss + classification_loss

        loss_log.update({
            "train_loss": total_loss,
            "classification_loss": classification_loss,
        })

        return loss_log

    def on_train_epoch_start(self) -> None:
        if self.current_epoch % self.cluster_loss_interval == 0 and self.current_epoch < self.cluster_stop_epoch:
            pretty_print("train epoch start (network.VGG16BN 106)")
            with torch.no_grad():
                total_feature_map = []
                for batch in self.train_dataloader():
                    inputs, _ = batch
                    inputs = inputs.to(self.device)
                    self.model(inputs)
                    total_feature_map.append(
                        self.intercept_output["5.3"].detach().cpu().numpy()
                    )
                total_feature_map = np.concatenate(total_feature_map, axis=0)
                sample, channel, height, width = total_feature_map.shape
                total_feature_map = total_feature_map.reshape(sample, channel, height * width)
                feature_map_correlation: np.ndarray = Similarity.pearson(total_feature_map, "numpy")
                feature_map_similarity = (feature_map_correlation + 1) / 2
                (
                    self.rough_cluster_mapping,
                    self.rough_intra_similarity_mask,
                    self.rough_inter_similarity_mask,
                ) = self.clustering.compute(
                    feature_map_similarity,
                    self.center_num
                )

    def conv_2d_filter(self, name: str, layer: nn.Module) -> tuple[bool, str]:
        return (  # ReLU
            isinstance(layer, nn.ReLU)
            and "model" not in name
            and "classifier" not in name,
            name.replace("features", "relu")
        )

        # return (  # conv
        #     isinstance(layer, nn.Conv2d)
        #     and "model" not in name,
        #     name.replace("features", "conv")
        # )

    def hook_feature_map(self, name, layer):
        def hook(module, input, output):
            # check lock, to avoid recursive hooking
            if self.epoch_log_lock:
                self.grid_images.update(get_sample_feature_map(name, layer, module, input, output))
            if name == "relu.42":
                self.total_filter_feature_map = get_total_feature_map(name, layer, module, input, output)

        return hook

    def log_tmp_output(self):
        self.epoch_log_lock = True  # lock on, to avoid recursive hooking
        with torch.no_grad():
            self.model(self.example_input.to(self.device))  # Forward pass using example input
        self.epoch_log_lock = False  # lock off

        contribution_norm = get_contribution_norms(
            self.intercept_output["5.2"],
            dict([*self.model.named_modules()])[self.total_filter_layer_name],
            self.device
        )
        contribution_fig = draw_contribution_norms(contribution_norm, "conv.5.3", "relu.5.2")
        save_contribution_norms(contribution_fig, self.logger, self.global_step)

        save_sample_feature_maps(self.example_input, self.grid_images, self.logger, self.global_step, 5)

    def on_train_end(self):
        if self.save_feature_map:
            save_total_feature_map(self.total_filter_layer_name, self.total_filter_feature_map)
