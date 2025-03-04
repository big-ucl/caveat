from typing import Union

import numpy as np
import torch
from torch import Tensor, tensor
from torch.optim.lr_scheduler import _LRScheduler

from caveat import current_device


def duration_mask(mask: Tensor) -> Tensor:
    duration_mask = mask.clone()
    duration_mask[:, 0] = 0.0
    duration_mask[
        torch.arange(duration_mask.shape[0]), (mask != 0).cumsum(-1).argmax(1)
    ] = 0.0
    return duration_mask


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
    patience: int = 20,
) -> int:
    """
    Calculate the output padding required for a 1D transposed convolution to achieve a target length.
    This function iterates over possible padding values and output padding values to find a combination
    that results in the desired target length after a 1D transposed convolution.
    Args:
        length (int): The length of the input.
        target (int): The desired length of the output.
        kernel_size (int): The size of the convolution kernel.
        stride (int): The stride of the convolution.
        padding (int): The initial padding value.
        patience (int, optional): The maximum number of iterations to try for padding and output padding. Default is 20.
    Returns:
        tuple: A tuple containing the padding and output padding values that achieve the target length.
    Raises:
        ValueError: If no combination of padding and output padding can achieve the target length within the given patience.
    """

    for pad in range(padding, padding + patience):
        for i in range(patience):
            if transconv_size_1d(length, kernel_size, stride, pad, i) == target:
                if pad != padding:
                    print(
                        f"Changed padding from {padding} to {pad} for target {target}."
                    )
                return pad, i
    raise ValueError(
        f"""Could not find input and output padding combination for target {target},
        length {length}, kernel_size {kernel_size}, stride {stride}, padding {padding}.
    """
    )


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


def build_hidden_layers(config: dict) -> list:
    """
    Build hidden layer sizes from config.

    Args:
        config (dict): Configuration dictionary containing hidden layer parameters.

    Raises:
        ValueError: If both hidden_layers and hidden_n/hidden_size are specified.
        ValueError: If hidden_layers is not a list.
        ValueError: If hidden_layers contains non-integer values.
        ValueError: If neither hidden_layers nor hidden_n/hidden_size are specified.

    Returns:
        list: List of hidden layer sizes.
    """
    hidden_layers = config.get("hidden_layers", None)
    hidden_n = config.get("hidden_n", None)
    hidden_size = config.get("hidden_size", None)
    if hidden_layers is not None:
        if hidden_n is not None or hidden_size is not None:
            raise ValueError(
                "Cannot specify hidden_layers and layer_n or layer_size"
            )
        if not isinstance(hidden_layers, list):
            raise ValueError("hidden_layers must be a list")
        for layer in hidden_layers:
            if not isinstance(layer, int):
                raise ValueError("hidden_layers must be a list of integers")
        return hidden_layers
    if hidden_n is not None and hidden_size is not None:
        return [int(hidden_size)] * int(hidden_n)
    raise ValueError("Must specify hidden_layers or layer_n and layer_size")
