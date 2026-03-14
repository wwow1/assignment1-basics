import torch
import numpy as np

def data_loading(dataset, batch_size: int, context_length: int, device: str):
    """
    Load the dataset into batches of sequences.

    Args:
        dataset (npt.NDArray): The dataset to load.
        batch_size (int): The number of sequences in each batch.
        context_length (int): The length of each sequence.
        device (str): The device to load the data onto.

    Returns:
        A tuple of (batches, labels), where batches is a tensor of shape (batch_size, context_length)
        and labels is a tensor of shape (batch_size, context_length).
    """
    limit = dataset.size - context_length
    indices = np.random.randint(0, limit, size=batch_size)

    src = torch.from_numpy(np.stack([dataset[i : i + context_length] for i in indices])).int().to(device)
    dst = torch.from_numpy(np.stack([dataset[i + 1 : i + context_length + 1] for i in indices])).int().to(device)
    return src, dst

