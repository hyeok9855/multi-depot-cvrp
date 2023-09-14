# """
# Utils for envs
# """
# from tensordict.tensordict import TensorDict
# import numpy as np
# import torch


# def npdict_to_tensordict(npdict: dict[str, np.ndarray], device: str = None) -> TensorDict:
#     batch_size = npdict[list(npdict.keys())[0]].shape[0]
#     return TensorDict(npdict, batch_size=batch_size, device=device)


def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example (with dim=1):
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    return src.gather(dim, idx).squeeze() if squeeze else src.gather(dim, idx)
