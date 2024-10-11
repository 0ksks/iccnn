import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def mapping2label(mapping: torch.Tensor) -> torch.Tensor:
    mapping = mapping.numpy(force=True)
    sparse_matrix = csr_matrix(mapping)
    _, labels = connected_components(sparse_matrix, directed=False)
    return torch.from_numpy(labels).long()
