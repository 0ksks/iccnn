from typing import Union, Literal
from torch.utils.data import DataLoader


def get_dataloader(
        datasets: Union[list, tuple],
        train_kwargs: dict = None,
        test_kwargs: dict = None,
        batch_size=32,
        split: Literal["train", "test"] = None
):
    if not train_kwargs:
        train_kwargs = {}
    if not test_kwargs:
        test_kwargs = {}
    if split == "train":
        train_dataloader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True, **train_kwargs)
        return train_dataloader
    elif split == "test":
        test_dataloader = DataLoader(datasets[0], batch_size=batch_size, shuffle=False, **test_kwargs)
        return test_dataloader
    else:
        train_dataloader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True, **train_kwargs)
        test_dataloader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False, **test_kwargs)
        return train_dataloader, test_dataloader
