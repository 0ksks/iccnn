import numpy as np
import torch

from pretty_print import pretty_print


class Clustering:
    def __init__(self, device: torch.device, random_state=21):
        self.cluster_class = None
        self.random_state = random_state
        self.device = device

    def set_cluster_class(self, cluster_class):
        self.cluster_class = cluster_class

    def compute(self, similarity_matrix: np.ndarray, cluster_centers: int):
        """
        return
        ground_truth(similarity_matrix.shape) (C, C)
        ground_truth[i][j] == 1 if C_i, C_j are in the same cluster, else 0

        intra_similarity_mask(cluster_centers, similarity_matrix.shape) (cluster_centers, C, C)

        inter_similarity_mask(cluster_centers, similarity_matrix.shape) (cluster_centers, C, C)
        """
        pretty_print("clustering... (network.util 92)")
        cluster_method = self.cluster_class(
            n_clusters=cluster_centers,
            affinity="precomputed",
            random_state=self.random_state
        )
        y_pred = cluster_method.fit_predict(similarity_matrix)
        ground_truth = np.zeros_like(similarity_matrix)
        intra_similarity_mask, inter_similarity_mask = [], []

        for center in range(cluster_centers):
            target_sample_index = np.where(y_pred == center)[0]
            current_intra_similarity_mask = np.zeros_like(similarity_matrix)
            current_inter_similarity_mask = np.zeros_like(similarity_matrix)
            for sample_index in target_sample_index:
                ground_truth[sample_index, target_sample_index] = 1
                current_intra_similarity_mask[sample_index, target_sample_index] = 1
                current_inter_similarity_mask[sample_index, :] = 1
            intra_similarity_mask.append(np.expand_dims(current_intra_similarity_mask, 0))
            inter_similarity_mask.append(np.expand_dims(current_inter_similarity_mask, 0))

        intra_similarity_mask = np.concatenate(intra_similarity_mask, axis=0)
        inter_similarity_mask = np.concatenate(inter_similarity_mask, axis=0)

        ground_truth = torch.from_numpy(ground_truth).float().to(self.device)
        intra_similarity_mask = torch.from_numpy(intra_similarity_mask).float().to(self.device)
        inter_similarity_mask = torch.from_numpy(inter_similarity_mask).float().to(self.device)
        return ground_truth, intra_similarity_mask, inter_similarity_mask
