import torch
import typing
import os

def save_checkpoint(
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer, 
    iteration : int, 
    filename : str | os.PathLike | typing.BinaryIO,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": iteration,
        },
        filename,
    )

def load_checkpoint(
    filename : str | os.PathLike | typing.BinaryIO,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer, 
):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["step"]
    return iteration
