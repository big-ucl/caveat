from typing import Union

import numpy as np
import torch
from torch import tensor
from torch.optim.lr_scheduler import _LRScheduler

from caveat import current_device


def hot_argmax(batch: tensor, axis: int = -1) -> tensor:
    """Encoded given axis as one-hot based on argmax for that axis.

    Args:
        batch (tensor): Input tensor.
        axis (int, optional): Axis index to encode. Defaults to -1.

    Returns:
        tensor: One hot encoded tensor.
    """
    batch = batch.swapaxes(axis, -1)
    argmax = batch.argmax(axis=-1)
    eye = torch.eye(batch.shape[-1])
    eye = eye.to(current_device())
    batch = eye[argmax]
    return batch.swapaxes(axis, -1)


def conv2d_size(
    size: Union[tuple[int, int], int],
    kernel_size: Union[tuple[int, int], int] = 3,
    stride: Union[tuple[int, int], int] = 2,
    padding: Union[tuple[int, int], int] = 1,
    dilation: Union[tuple[int, int], int] = 1,
) -> np.array:
    """Calculate output dimensions for 2d convolution.

    Args:
        size (Union[tuple[int, int], int]): Input size, may be integer if symetric.
        kernel_size (Union[tuple[int, int], int], optional): Kernel_size. Defaults to 3.
        stride (Union[tuple[int, int], int], optional): Stride. Defaults to 2.
        padding (Union[tuple[int, int], int], optional): Input padding. Defaults to 1.
        dilation (Union[tuple[int, int], int], optional): Dilation. Defaults to 1.

    Returns:
        np.array: Output size.
    """
    if isinstance(size, int):
        size = (size, size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    return (
        np.array(size)
        + 2 * np.array(padding)
        - np.array(dilation) * (np.array(kernel_size) - 1)
        - 1
    ) // np.array(stride) + 1


def conv1d_size(
    length: int, kernel_size: int, stride: int, padding: int = 0
) -> int:
    """Calculate output dimensions for 1d convolution.

    Args:
        length (int): Input size.
        kernel_size (int): Kernel_size.
        stride (int): Stride.
        padding (int): Input padding.
    Returns:
        int: Output size.
    """
    return int((length - (kernel_size - 1) + (2 * padding) - 1) / stride) + 1


def transconv_size_2d(
    size: Union[tuple[int, int], int],
    kernel_size: Union[tuple[int, int], int] = 3,
    stride: Union[tuple[int, int], int] = 2,
    padding: Union[tuple[int, int], int] = 1,
    dilation: Union[tuple[int, int], int] = 1,
    output_padding: Union[tuple[int, int], int] = 1,
) -> np.array:
    """Calculate output dimension for 2d transpose convolution.

    Args:
        size (Union[tuple[int, int], int]): Input size, may be integer if symetric.
        kernel_size (Union[tuple[int, int], int], optional): Kernel size. Defaults to 3.
        stride (Union[tuple[int, int], int], optional): Stride. Defaults to 2.
        padding (Union[tuple[int, int], int], optional): Input padding. Defaults to 1.
        dilation (Union[tuple[int, int], int], optional): Dilation. Defaults to 1.
        output_padding (Union[tuple[int, int], int], optional): Output padding. Defaults to 1.

    Returns:
        np.array: Output size.
    """
    if isinstance(size, int):
        size = (size, size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    return (
        (np.array(size) - 1) * np.array(stride)
        - 2 * np.array(padding)
        + np.array(dilation) * (np.array(kernel_size) - 1)
        + np.array(output_padding)
        + 1
    )


def transconv_size_1d(
    length, kernel_size, stride, padding, output_padding, dilation=1
):
    return (
        (length - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


def calc_output_padding_1d(
    length: int,
    target: int,
    kernel_size: int,
    stride: int,
    padding: int,
    patience: int = 10,
) -> int:
    for i in range(patience):
        if transconv_size_1d(length, kernel_size, stride, padding, i) == target:
            return i
    raise ValueError(f"Could not find output padding for target {target}")


def calc_output_padding_2d(size: Union[tuple[int, int, int], int]) -> np.array:
    """Calculate output padding for a transposed convolution such that output dims will
    match dimensions of inputs to a convolution of given size.
    For each dimension, padding is set to 1 if even size, otherwise 0.

    Args:
        size (Union[tuple[int, int, int], int]): input size (h, w)

    Returns:
        np.array: required padding
    """
    if isinstance(size, int):
        size = (0, size, size)
    _, h, w = size
    return (int(h % 2 == 0), int(w % 2 == 0))


# class LearningRateScheduler(_LRScheduler):

#     def __init__(self, optimizer, init_lr):
#         self.optimizer = optimizer
#         self.init_lr = init_lr

#     def step(self, *args, **kwargs):
#         raise NotImplementedError

#     @staticmethod
#     def set_lr(optimizer, lr):
#         for g in optimizer.param_groups:
#             g["lr"] = lr

#     def get_lr(self):
#         for g in self.optimizer.param_groups:
#             return g["lr"]


class ScheduledOptim(_LRScheduler):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model**-0.5) * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )

    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
