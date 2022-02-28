# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of "vanilla" transforms for utility functions
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import logging
import sys
import time
import warnings
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Randomizable, RandomizableTransform, Transform
from monai.transforms.utils import (
    extreme_points_to_image,
    get_extreme_points,
    map_binary_to_indices,
    map_classes_to_indices,
)
from monai.transforms.utils_pytorch_numpy_unification import concatenate, in1d, moveaxis, unravel_indices
from monai.utils import (
    convert_data_type,
    convert_to_cupy,
    convert_to_numpy,
    convert_to_tensor,
    deprecated_arg,
    ensure_tuple,
    look_up_option,
    min_version,
    optional_import,
)
from monai.utils.enums import TransformBackends
from monai.utils.misc import is_module_ver_at_least
from monai.utils.type_conversion import convert_to_dst_type, get_equivalent_dtype

PILImageImage, has_pil = optional_import("PIL.Image", name="Image")
pil_image_fromarray, _ = optional_import("PIL.Image", name="fromarray")
cp, has_cp = optional_import("cupy")


__all__ = [
    "Identity",
    "AsChannelFirst",
    "AsChannelLast",
    "AddChannel",
    "AddCoordinateChannels",
    "EnsureChannelFirst",
    "EnsureType",
    "RepeatChannel",
    "RemoveRepeatedChannel",
    "SplitChannel",
    "CastToType",
    "ToTensor",
    "ToNumpy",
    "ToPIL",
    "Transpose",
    "SqueezeDim",
    "DataStats",
    "SimulateDelay",
    "Lambda",
    "RandLambda",
    "LabelToMask",
    "FgBgToIndices",
    "ClassesToIndices",
    "ConvertToMultiChannelBasedOnBratsClasses",
    "AddExtremePointsChannel",
    "TorchVision",
    "MapLabelValue",
    "IntensityStats",
    "ToDevice",
    "CuCIM",
    "RandCuCIM",
    "ToCupy",
]


class Identity(Transform):
    """
    Do nothing to the data.
    As the output value is same as input, it can be used as a testing tool to verify the transform chain,
    Compose or transform adaptor, etc.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        return img


class AsChannelFirst(Transform):
    """
    Change the channel dimension of the image to the first dimension.

    Most of the image transformations in ``monai.transforms``
    assume the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used to convert, for example, a channel-last image array in shape
    (spatial_dim_1[, spatial_dim_2, ...], num_channels) into the channel-first format,
    so that the multidimensional image array can be correctly interpreted by the other transforms.

    Args:
        channel_dim: which dimension of input image is the channel, default is the last dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, channel_dim: int = -1) -> None:
        if not (isinstance(channel_dim, int) and channel_dim >= -1):
            raise AssertionError("invalid channel dimension.")
        self.channel_dim = channel_dim

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        return moveaxis(img, self.channel_dim, 0)


class AsChannelLast(Transform):
    """
    Change the channel dimension of the image to the last dimension.

    Some of other 3rd party transforms assume the input image is in the channel-last format with shape
    (spatial_dim_1[, spatial_dim_2, ...], num_channels).

    This transform could be used to convert, for example, a channel-first image array in shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]) into the channel-last format,
    so that MONAI transforms can construct a chain with other 3rd party transforms together.

    Args:
        channel_dim: which dimension of input image is the channel, default is the first dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, channel_dim: int = 0) -> None:
        if not (isinstance(channel_dim, int) and channel_dim >= -1):
            raise AssertionError("invalid channel dimension.")
        self.channel_dim = channel_dim

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        return moveaxis(img, self.channel_dim, -1)


class AddChannel(Transform):
    """
    Adds a 1-length channel dimension to the input image.

    Most of the image transformations in ``monai.transforms``
    assumes the input image is in the channel-first format, which has the shape
    (num_channels, spatial_dim_1[, spatial_dim_2, ...]).

    This transform could be used, for example, to convert a (spatial_dim_1[, spatial_dim_2, ...])
    spatial image into the channel-first format so that the
    multidimensional image array can be correctly interpreted by the other
    transforms.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        return img[None]


class EnsureChannelFirst(Transform):
    """
    Automatically adjust or add the channel dimension of input data to ensure `channel_first` shape.
    It extracts the `original_channel_dim` info from provided meta_data dictionary.
    Typical values of `original_channel_dim` can be: "no_channel", 0, -1.
    Convert the data to `channel_first` based on the `original_channel_dim` information.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, strict_check: bool = True):
        """
        Args:
            strict_check: whether to raise an error when the meta information is insufficient.
        """
        self.strict_check = strict_check

    def __call__(self, img: NdarrayOrTensor, meta_dict: Optional[Mapping] = None) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if not isinstance(meta_dict, Mapping):
            msg = "meta_dict not available, EnsureChannelFirst is not in use."
            if self.strict_check:
                raise ValueError(msg)
            warnings.warn(msg)
            return img

        channel_dim = meta_dict.get("original_channel_dim")

        if channel_dim is None:
            msg = "Unknown original_channel_dim in the meta_dict, EnsureChannelFirst is not in use."
            if self.strict_check:
                raise ValueError(msg)
            warnings.warn(msg)
            return img
        if channel_dim == "no_channel":
            return AddChannel()(img)
        return AsChannelFirst(channel_dim=channel_dim)(img)


class RepeatChannel(Transform):
    """
    Repeat channel data to construct expected input shape for models.
    The `repeats` count includes the origin data, for example:
    ``RepeatChannel(repeats=2)([[1, 2], [3, 4]])`` generates: ``[[1, 2], [1, 2], [3, 4], [3, 4]]``

    Args:
        repeats: the number of repetitions for each element.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, repeats: int) -> None:
        if repeats <= 0:
            raise AssertionError("repeats count must be greater than 0.")
        self.repeats = repeats

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a "channel-first" array.
        """
        repeat_fn = torch.repeat_interleave if isinstance(img, torch.Tensor) else np.repeat
        return repeat_fn(img, self.repeats, 0)  # type: ignore


class RemoveRepeatedChannel(Transform):
    """
    RemoveRepeatedChannel data to undo RepeatChannel
    The `repeats` count specifies the deletion of the origin data, for example:
    ``RemoveRepeatedChannel(repeats=2)([[1, 2], [1, 2], [3, 4], [3, 4]])`` generates: ``[[1, 2], [3, 4]]``

    Args:
        repeats: the number of repetitions to be deleted for each element.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, repeats: int) -> None:
        if repeats <= 0:
            raise AssertionError("repeats count must be greater than 0.")

        self.repeats = repeats

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a "channel-first" array.
        """
        if img.shape[0] < 2:
            raise AssertionError("Image must have more than one channel")

        return img[:: self.repeats, :]


class SplitChannel(Transform):
    """
    Split Numpy array or PyTorch Tensor data according to the channel dim.
    It can help applying different following transforms to different channels.

    Args:
        channel_dim: which dimension of input image is the channel, default to 0.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, channel_dim: int = 0) -> None:
        self.channel_dim = channel_dim

    def __call__(self, img: NdarrayOrTensor) -> List[NdarrayOrTensor]:
        num_classes = img.shape[self.channel_dim]
        if num_classes <= 1:
            raise RuntimeError("input image does not contain multiple channels.")

        outputs = []
        slices = [slice(None)] * len(img.shape)
        for i in range(num_classes):
            slices[self.channel_dim] = slice(i, i + 1)
            outputs.append(img[tuple(slices)])

        return outputs


class CastToType(Transform):
    """
    Cast the Numpy data to specified numpy data type, or cast the PyTorch Tensor to
    specified PyTorch data type.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dtype=np.float32) -> None:
        """
        Args:
            dtype: convert image to this data type, default is `np.float32`.
        """
        self.dtype = dtype

    def __call__(self, img: NdarrayOrTensor, dtype: Optional[Union[DtypeLike, torch.dtype]] = None) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a numpy array or PyTorch Tensor.

        Args:
            dtype: convert image to this data type, default is `self.dtype`.

        Raises:
            TypeError: When ``img`` type is not in ``Union[numpy.ndarray, torch.Tensor]``.

        """
        img_out, *_ = convert_data_type(img, output_type=type(img), dtype=dtype or self.dtype)
        return img_out


class ToTensor(Transform):
    """
    Converts the input image to a tensor without applying any other transformations.
    Input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
    Will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
    For dictionary, list or tuple, convert every item to a Tensor if applicable and `wrap_sequence=False`.

    Args:
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
            E.g., if `False`, `[1, 2]` -> `[tensor(1), tensor(2)]`, if `True`, then `[1, 2]` -> `tensor([1, 2])`.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None, wrap_sequence: bool = True
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.wrap_sequence = wrap_sequence

    def __call__(self, img: NdarrayOrTensor):
        """
        Apply the transform to `img` and make it contiguous.
        """
        return convert_to_tensor(img, dtype=self.dtype, device=self.device, wrap_sequence=self.wrap_sequence)


class EnsureType(Transform):
    """
    Ensure the input data to be a PyTorch Tensor or numpy array, support: `numpy array`, `PyTorch Tensor`,
    `float`, `int`, `bool`, `string` and `object` keep the original.
    If passing a dictionary, list or tuple, still return dictionary, list or tuple will recursively convert
    every item to the expected data type if `wrap_sequence=False`.

    Args:
        data_type: target data type to convert, should be "tensor" or "numpy".
        dtype: target data content type to convert, for example: np.float32, torch.float, etc.
        device: for Tensor data type, specify the target device.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
            E.g., if `False`, `[1, 2]` -> `[tensor(1), tensor(2)]`, if `True`, then `[1, 2]` -> `tensor([1, 2])`.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        data_type: str = "tensor",
        dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
        device: Optional[torch.device] = None,
        wrap_sequence: bool = True,
    ) -> None:
        self.data_type = look_up_option(data_type.lower(), {"tensor", "numpy"})
        self.dtype = dtype
        self.device = device
        self.wrap_sequence = wrap_sequence

    def __call__(self, data: NdarrayOrTensor):
        """
        Args:
            data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
                will ensure Tensor, Numpy array, float, int, bool as Tensors or numpy arrays, strings and
                objects keep the original. for dictionary, list or tuple, ensure every item as expected type
                if applicable and `wrap_sequence=False`.

        """
        output_type = torch.Tensor if self.data_type == "tensor" else np.ndarray
        out, *_ = convert_data_type(
            data=data, output_type=output_type, dtype=self.dtype, device=self.device, wrap_sequence=self.wrap_sequence
        )
        return out


class ToNumpy(Transform):
    """
    Converts the input data to numpy array, can support list or tuple of numbers and PyTorch Tensor.

    Args:
        dtype: target data type when converting to numpy array.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
            E.g., if `False`, `[1, 2]` -> `[array(1), array(2)]`, if `True`, then `[1, 2]` -> `array([1, 2])`.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dtype: DtypeLike = None, wrap_sequence: bool = True) -> None:
        super().__init__()
        self.dtype = dtype
        self.wrap_sequence = wrap_sequence

    def __call__(self, img: NdarrayOrTensor):
        """
        Apply the transform to `img` and make it contiguous.
        """
        return convert_to_numpy(img, dtype=self.dtype, wrap_sequence=self.wrap_sequence)


class ToCupy(Transform):
    """
    Converts the input data to CuPy array, can support list or tuple of numbers, NumPy and PyTorch Tensor.

    Args:
        dtype: data type specifier. It is inferred from the input by default.
            if not None, must be an argument of `numpy.dtype`, for more details:
            https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html.
        wrap_sequence: if `False`, then lists will recursively call this function, default to `True`.
            E.g., if `False`, `[1, 2]` -> `[array(1), array(2)]`, if `True`, then `[1, 2]` -> `array([1, 2])`.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dtype: Optional[np.dtype] = None, wrap_sequence: bool = True) -> None:
        super().__init__()
        self.dtype = dtype
        self.wrap_sequence = wrap_sequence

    def __call__(self, data: NdarrayOrTensor):
        """
        Create a CuPy array from `data` and make it contiguous
        """
        return convert_to_cupy(data, dtype=self.dtype, wrap_sequence=self.wrap_sequence)


class ToPIL(Transform):
    """
    Converts the input image (in the form of NumPy array or PyTorch Tensor) to PIL image
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """
        if isinstance(img, PILImageImage):
            return img
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        return pil_image_fromarray(img)


class Transpose(Transform):
    """
    Transposes the input image based on the given `indices` dimension ordering.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, indices: Optional[Sequence[int]]) -> None:
        self.indices = None if indices is None else tuple(indices)

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        if isinstance(img, torch.Tensor):
            return img.permute(self.indices or tuple(range(img.ndim)[::-1]))
        return img.transpose(self.indices)  # type: ignore


class SqueezeDim(Transform):
    """
    Squeeze a unitary dimension.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, dim: Optional[int] = 0) -> None:
        """
        Args:
            dim: dimension to be squeezed. Default = 0
                "None" works when the input is numpy array.

        Raises:
            TypeError: When ``dim`` is not an ``Optional[int]``.

        """
        if dim is not None and not isinstance(dim, int):
            raise TypeError(f"dim must be None or a int but is {type(dim).__name__}.")
        self.dim = dim

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: numpy arrays with required dimension `dim` removed
        """
        if self.dim is None:
            return img.squeeze()
        # for pytorch/numpy unification
        if img.shape[self.dim] != 1:
            raise ValueError("Can only squeeze singleton dimension")
        return img.squeeze(self.dim)


class DataStats(Transform):
    """
    Utility transform to show the statistics of data for debug or analysis.
    It can be inserted into any place of a transform chain and check results of previous transforms.
    It support both `numpy.ndarray` and `torch.tensor` as input data,
    so it can be used in pre-processing and post-processing.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        prefix: str = "Data",
        data_type: bool = True,
        data_shape: bool = True,
        value_range: bool = True,
        data_value: bool = False,
        additional_info: Optional[Callable] = None,
        logger_handler: Optional[logging.Handler] = None,
    ) -> None:
        """
        Args:
            prefix: will be printed in format: "{prefix} statistics".
            data_type: whether to show the type of input data.
            data_shape: whether to show the shape of input data.
            value_range: whether to show the value range of input data.
            data_value: whether to show the raw value of input data.
                a typical example is to print some properties of Nifti image: affine, pixdim, etc.
            additional_info: user can define callable function to extract additional info from input data.
            logger_handler: add additional handler to output data: save to file, etc.
                all the existing python logging handlers: https://docs.python.org/3/library/logging.handlers.html.
                the handler should have a logging level of at least `INFO`.

        Raises:
            TypeError: When ``additional_info`` is not an ``Optional[Callable]``.

        """
        if not isinstance(prefix, str):
            raise AssertionError("prefix must be a string.")
        self.prefix = prefix
        self.data_type = data_type
        self.data_shape = data_shape
        self.value_range = value_range
        self.data_value = data_value
        if additional_info is not None and not callable(additional_info):
            raise TypeError(f"additional_info must be None or callable but is {type(additional_info).__name__}.")
        self.additional_info = additional_info
        self._logger_name = "DataStats"
        _logger = logging.getLogger(self._logger_name)
        _logger.setLevel(logging.INFO)
        console = logging.StreamHandler(sys.stdout)  # always stdout
        console.setLevel(logging.INFO)
        _logger.addHandler(console)
        if logger_handler is not None:
            _logger.addHandler(logger_handler)

    def __call__(
        self,
        img: NdarrayOrTensor,
        prefix: Optional[str] = None,
        data_type: Optional[bool] = None,
        data_shape: Optional[bool] = None,
        value_range: Optional[bool] = None,
        data_value: Optional[bool] = None,
        additional_info: Optional[Callable] = None,
    ) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, optionally take arguments similar to the class constructor.
        """
        lines = [f"{prefix or self.prefix} statistics:"]

        if self.data_type if data_type is None else data_type:
            lines.append(f"Type: {type(img)} {img.dtype if hasattr(img, 'dtype') else None}")
        if self.data_shape if data_shape is None else data_shape:
            lines.append(f"Shape: {img.shape}")
        if self.value_range if value_range is None else value_range:
            if isinstance(img, np.ndarray):
                lines.append(f"Value range: ({np.min(img)}, {np.max(img)})")
            elif isinstance(img, torch.Tensor):
                lines.append(f"Value range: ({torch.min(img)}, {torch.max(img)})")
            else:
                lines.append(f"Value range: (not a PyTorch or Numpy array, type: {type(img)})")
        if self.data_value if data_value is None else data_value:
            lines.append(f"Value: {img}")
        additional_info = self.additional_info if additional_info is None else additional_info
        if additional_info is not None:
            lines.append(f"Additional info: {additional_info(img)}")
        separator = "\n"
        output = f"{separator.join(lines)}"
        logging.getLogger(self._logger_name).info(output)
        return img


class SimulateDelay(Transform):
    """
    This is a pass through transform to be used for testing purposes. It allows
    adding fake behaviors that are useful for testing purposes to simulate
    how large datasets behave without needing to test on large data sets.

    For example, simulating slow NFS data transfers, or slow network transfers
    in testing by adding explicit timing delays. Testing of small test data
    can lead to incomplete understanding of real world issues, and may lead
    to sub-optimal design choices.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, delay_time: float = 0.0) -> None:
        """
        Args:
            delay_time: The minimum amount of time, in fractions of seconds,
                to accomplish this delay task.
        """
        super().__init__()
        self.delay_time: float = delay_time

    def __call__(self, img: NdarrayOrTensor, delay_time: Optional[float] = None) -> NdarrayOrTensor:
        """
        Args:
            img: data remain unchanged throughout this transform.
            delay_time: The minimum amount of time, in fractions of seconds,
                to accomplish this delay task.
        """
        time.sleep(self.delay_time if delay_time is None else delay_time)
        return img


class Lambda(Transform):
    """
    Apply a user-defined lambda as a transform.

    For example:

    .. code-block:: python
        :emphasize-lines: 2

        image = np.ones((10, 2, 2))
        lambd = Lambda(func=lambda x: x[:4, :, :])
        print(lambd(image).shape)
        (4, 2, 2)

    Args:
        func: Lambda/function to be applied.

    Raises:
        TypeError: When ``func`` is not an ``Optional[Callable]``.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, func: Optional[Callable] = None) -> None:
        if func is not None and not callable(func):
            raise TypeError(f"func must be None or callable but is {type(func).__name__}.")
        self.func = func

    def __call__(self, img: NdarrayOrTensor, func: Optional[Callable] = None):
        """
        Apply `self.func` to `img`.

        Args:
            func: Lambda/function to be applied. Defaults to `self.func`.

        Raises:
            TypeError: When ``func`` is not an ``Optional[Callable]``.
            ValueError: When ``func=None`` and ``self.func=None``. Incompatible values.

        """
        if func is not None:
            if not callable(func):
                raise TypeError(f"func must be None or callable but is {type(func).__name__}.")
            return func(img)
        if self.func is not None:
            return self.func(img)
        raise ValueError("Incompatible values: func=None and self.func=None.")


class RandLambda(Lambda, RandomizableTransform):
    """
    Randomizable version :py:class:`monai.transforms.Lambda`, the input `func` may contain random logic,
    or randomly execute the function based on `prob`.

    Args:
        func: Lambda/function to be applied.
        prob: probability of executing the random function, default to 1.0, with 100% probability to execute.

    For more details, please check :py:class:`monai.transforms.Lambda`.
    """

    backend = Lambda.backend

    def __init__(self, func: Optional[Callable] = None, prob: float = 1.0) -> None:
        Lambda.__init__(self=self, func=func)
        RandomizableTransform.__init__(self=self, prob=prob)

    def __call__(self, img: NdarrayOrTensor, func: Optional[Callable] = None):
        self.randomize(img)
        return super().__call__(img=img, func=func) if self._do_transform else img


class LabelToMask(Transform):
    """
    Convert labels to mask for other tasks. A typical usage is to convert segmentation labels
    to mask data to pre-process images and then feed the images into classification network.
    It can support single channel labels or One-Hot labels with specified `select_labels`.
    For example, users can select `label value = [2, 3]` to construct mask data, or select the
    second and the third channels of labels to construct mask data.
    The output mask data can be a multiple channels binary data or a single channel binary
    data that merges all the channels.

    Args:
        select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
            is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
            `select_labels` is the expected channel indices.
        merge_channels: whether to use `np.any()` to merge the result on channel dim. if yes,
            will return a single channel mask with binary data.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(  # pytype: disable=annotation-type-mismatch
        self, select_labels: Union[Sequence[int], int], merge_channels: bool = False
    ) -> None:  # pytype: disable=annotation-type-mismatch
        self.select_labels = ensure_tuple(select_labels)
        self.merge_channels = merge_channels

    def __call__(
        self,
        img: NdarrayOrTensor,
        select_labels: Optional[Union[Sequence[int], int]] = None,
        merge_channels: bool = False,
    ) -> NdarrayOrTensor:
        """
        Args:
            select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
                is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
                `select_labels` is the expected channel indices.
            merge_channels: whether to use `np.any()` to merge the result on channel dim. if yes,
                will return a single channel mask with binary data.
        """
        if select_labels is None:
            select_labels = self.select_labels
        else:
            select_labels = ensure_tuple(select_labels)

        if img.shape[0] > 1:
            data = img[[*select_labels]]
        else:
            where: Callable = np.where if isinstance(img, np.ndarray) else torch.where  # type: ignore
            if isinstance(img, np.ndarray) or is_module_ver_at_least(torch, (1, 8, 0)):
                data = where(in1d(img, select_labels), True, False).reshape(img.shape)
            # pre pytorch 1.8.0, need to use 1/0 instead of True/False
            else:
                data = where(
                    in1d(img, select_labels), torch.tensor(1, device=img.device), torch.tensor(0, device=img.device)
                ).reshape(img.shape)

        if merge_channels or self.merge_channels:
            if isinstance(img, np.ndarray) or is_module_ver_at_least(torch, (1, 8, 0)):
                return data.any(0)[None]
            # pre pytorch 1.8.0 compatibility
            return data.to(torch.uint8).any(0)[None].to(bool)  # type: ignore

        return data


class FgBgToIndices(Transform):
    """
    Compute foreground and background of the input label data, return the indices.
    If no output_shape specified, output data will be 1 dim indices after flattening.
    This transform can help pre-compute foreground and background regions for other transforms.
    A typical usage is to randomly select foreground and background to crop.
    The main logic is based on :py:class:`monai.transforms.utils.map_binary_to_indices`.

    Args:
        image_threshold: if enabled `image` at runtime, use ``image > image_threshold`` to
            determine the valid image content area and select background only in this area.
        output_shape: expected shape of output indices. if not None, unravel indices to specified shape.

    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, image_threshold: float = 0.0, output_shape: Optional[Sequence[int]] = None) -> None:
        self.image_threshold = image_threshold
        self.output_shape = output_shape

    def __call__(
        self,
        label: NdarrayOrTensor,
        image: Optional[NdarrayOrTensor] = None,
        output_shape: Optional[Sequence[int]] = None,
    ) -> Tuple[NdarrayOrTensor, NdarrayOrTensor]:
        """
        Args:
            label: input data to compute foreground and background indices.
            image: if image is not None, use ``label = 0 & image > image_threshold``
                to define background. so the output items will not map to all the voxels in the label.
            output_shape: expected shape of output indices. if None, use `self.output_shape` instead.

        """
        if output_shape is None:
            output_shape = self.output_shape
        fg_indices, bg_indices = map_binary_to_indices(label, image, self.image_threshold)
        if output_shape is not None:
            fg_indices = unravel_indices(fg_indices, output_shape)
            bg_indices = unravel_indices(bg_indices, output_shape)
        return fg_indices, bg_indices


class ClassesToIndices(Transform):

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        num_classes: Optional[int] = None,
        image_threshold: float = 0.0,
        output_shape: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Compute indices of every class of the input label data, return a list of indices.
        If no output_shape specified, output data will be 1 dim indices after flattening.
        This transform can help pre-compute indices of the class regions for other transforms.
        A typical usage is to randomly select indices of classes to crop.
        The main logic is based on :py:class:`monai.transforms.utils.map_classes_to_indices`.

        Args:
            num_classes: number of classes for argmax label, not necessary for One-Hot label.
            image_threshold: if enabled `image` at runtime, use ``image > image_threshold`` to
                determine the valid image content area and select only the indices of classes in this area.
            output_shape: expected shape of output indices. if not None, unravel indices to specified shape.

        """
        self.num_classes = num_classes
        self.image_threshold = image_threshold
        self.output_shape = output_shape

    def __call__(
        self,
        label: NdarrayOrTensor,
        image: Optional[NdarrayOrTensor] = None,
        output_shape: Optional[Sequence[int]] = None,
    ) -> List[NdarrayOrTensor]:
        """
        Args:
            label: input data to compute the indices of every class.
            image: if image is not None, use ``image > image_threshold`` to define valid region, and only select
                the indices within the valid region.
            output_shape: expected shape of output indices. if None, use `self.output_shape` instead.

        """

        if output_shape is None:
            output_shape = self.output_shape
        indices: List[NdarrayOrTensor]
        indices = map_classes_to_indices(label, self.num_classes, image, self.image_threshold)
        if output_shape is not None:
            indices = [unravel_indices(cls_indices, output_shape) for cls_indices in indices]

        return indices


class ConvertToMultiChannelBasedOnBratsClasses(Transform):
    """
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 1) | (img == 4), (img == 1) | (img == 4) | (img == 2), img == 4]
        # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
        # label 4 is ET
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class AddExtremePointsChannel(Randomizable, Transform):
    """
    Add extreme points of label to the image as a new channel. This transform generates extreme
    point from label and applies a gaussian filter. The pixel values in points image are rescaled
    to range [rescale_min, rescale_max] and added as a new channel to input image. The algorithm is
    described in Roth et al., Going to Extremes: Weakly Supervised Medical Image Segmentation
    https://arxiv.org/abs/2009.11988.

    This transform only supports single channel labels (1, spatial_dim1, [spatial_dim2, ...]). The
    background ``index`` is ignored when calculating extreme points.

    Args:
        background: Class index of background label, defaults to 0.
        pert: Random perturbation amount to add to the points, defaults to 0.0.

    Raises:
        ValueError: When no label image provided.
        ValueError: When label image is not single channel.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, background: int = 0, pert: float = 0.0) -> None:
        self._background = background
        self._pert = pert
        self._points: List[Tuple[int, ...]] = []

    def randomize(self, label: NdarrayOrTensor) -> None:
        self._points = get_extreme_points(label, rand_state=self.R, background=self._background, pert=self._pert)

    def __call__(
        self,
        img: NdarrayOrTensor,
        label: Optional[NdarrayOrTensor] = None,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 3.0,
        rescale_min: float = -1.0,
        rescale_max: float = 1.0,
    ) -> NdarrayOrTensor:
        """
        Args:
            img: the image that we want to add new channel to.
            label: label image to get extreme points from. Shape must be
                (1, spatial_dim1, [, spatial_dim2, ...]). Doesn't support one-hot labels.
            sigma: if a list of values, must match the count of spatial dimensions of input data,
                and apply every value in the list to 1 spatial dimension. if only 1 value provided,
                use it for all spatial dimensions.
            rescale_min: minimum value of output data.
            rescale_max: maximum value of output data.
        """
        if label is None:
            raise ValueError("This transform requires a label array!")
        if label.shape[0] != 1:
            raise ValueError("Only supports single channel labels!")

        # Generate extreme points
        self.randomize(label[0, :])

        points_image = extreme_points_to_image(
            points=self._points, label=label, sigma=sigma, rescale_min=rescale_min, rescale_max=rescale_max
        )
        points_image, *_ = convert_to_dst_type(points_image, img)  # type: ignore
        return concatenate((img, points_image), axis=0)


class TorchVision:
    """
    This is a wrapper transform for PyTorch TorchVision transform based on the specified transform name and args.
    As most of the TorchVision transforms only work for PIL image and PyTorch Tensor, this transform expects input
    data to be PyTorch Tensor, users can easily call `ToTensor` transform to convert a Numpy array to Tensor.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, name: str, *args, **kwargs) -> None:
        """
        Args:
            name: The transform name in TorchVision package.
            args: parameters for the TorchVision transform.
            kwargs: parameters for the TorchVision transform.

        """
        super().__init__()
        self.name = name
        transform, _ = optional_import("torchvision.transforms", "0.8.0", min_version, name=name)
        self.trans = transform(*args, **kwargs)

    def __call__(self, img: NdarrayOrTensor):
        """
        Args:
            img: PyTorch Tensor data for the TorchVision transform.

        """
        img_t, *_ = convert_data_type(img, torch.Tensor)  # type: ignore
        out = self.trans(img_t)
        out, *_ = convert_to_dst_type(src=out, dst=img)
        return out


class MapLabelValue:
    """
    Utility to map label values to another set of values.
    For example, map [3, 2, 1] to [0, 1, 2], [1, 2, 3] -> [0.5, 1.5, 2.5], ["label3", "label2", "label1"] -> [0, 1, 2],
    [3.5, 2.5, 1.5] -> ["label0", "label1", "label2"], etc.
    The label data must be numpy array or array-like data and the output data will be numpy array.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, orig_labels: Sequence, target_labels: Sequence, dtype: DtypeLike = np.float32) -> None:
        """
        Args:
            orig_labels: original labels that map to others.
            target_labels: expected label values, 1: 1 map to the `orig_labels`.
            dtype: convert the output data to dtype, default to float32.

        """
        if len(orig_labels) != len(target_labels):
            raise ValueError("orig_labels and target_labels must have the same length.")
        if all(o == z for o, z in zip(orig_labels, target_labels)):
            raise ValueError("orig_labels and target_labels are exactly the same, should be different to map.")

        self.orig_labels = orig_labels
        self.target_labels = target_labels
        self.dtype = get_equivalent_dtype(dtype, data_type=np.ndarray)

    def __call__(self, img: NdarrayOrTensor):
        img_np, *_ = convert_data_type(img, np.ndarray)
        img_flat = img_np.flatten()
        try:
            out_flat = np.array(img_flat, dtype=self.dtype)
        except ValueError:
            # can't copy unchanged labels as the expected dtype is not supported, must map all the label values
            out_flat = np.zeros(shape=img_flat.shape, dtype=self.dtype)

        for o, t in zip(self.orig_labels, self.target_labels):
            if o == t:
                continue
            np.place(out_flat, img_flat == o, t)

        reshaped = out_flat.reshape(img_np.shape)
        out, *_ = convert_to_dst_type(src=reshaped, dst=img, dtype=self.dtype)
        return out


class IntensityStats(Transform):
    """
    Compute statistics for the intensity values of input image and store into the meta data dictionary.
    For example: if `ops=[lambda x: np.mean(x), "max"]` and `key_prefix="orig"`, may generate below stats:
    `{"orig_custom_0": 1.5, "orig_max": 3.0}`.

    Args:
        ops: expected operations to compute statistics for the intensity.
            if a string, will map to the predefined operations, supported: ["mean", "median", "max", "min", "std"]
            mapping to `np.nanmean`, `np.nanmedian`, `np.nanmax`, `np.nanmin`, `np.nanstd`.
            if a callable function, will execute the function on input image.
        key_prefix: the prefix to combine with `ops` name to generate the key to store the results in the
            meta data dictionary. if some `ops` are callable functions, will use "{key_prefix}_custom_{index}"
            as the key, where index counts from 0.
        channel_wise: whether to compute statistics for every channel of input image separately.
            if True, return a list of values for every operation, default to False.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, ops: Sequence[Union[str, Callable]], key_prefix: str, channel_wise: bool = False) -> None:
        self.ops = ensure_tuple(ops)
        self.key_prefix = key_prefix
        self.channel_wise = channel_wise

    def __call__(
        self, img: NdarrayOrTensor, meta_data: Optional[Dict] = None, mask: Optional[np.ndarray] = None
    ) -> Tuple[NdarrayOrTensor, Dict]:
        """
        Compute statistics for the intensity of input image.

        Args:
            img: input image to compute intensity stats.
            meta_data: meta data dictionary to store the statistics data, if None, will create an empty dictionary.
            mask: if not None, mask the image to extract only the interested area to compute statistics.
                mask must have the same shape as input `img`.

        """
        img_np: np.ndarray
        img_np, *_ = convert_data_type(img, np.ndarray)  # type: ignore
        if meta_data is None:
            meta_data = {}

        if mask is not None:
            if mask.shape != img_np.shape or mask.dtype != bool:
                raise TypeError("mask must be bool array with the same shape as input `img`.")
            img_np = img_np[mask]

        supported_ops = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "max": np.nanmax,
            "min": np.nanmin,
            "std": np.nanstd,
        }

        def _compute(op: Callable, data: np.ndarray):
            if self.channel_wise:
                return [op(c) for c in data]
            return op(data)

        custom_index = 0
        for o in self.ops:
            if isinstance(o, str):
                o = look_up_option(o, supported_ops.keys())
                meta_data[self.key_prefix + "_" + o] = _compute(supported_ops[o], img_np)  # type: ignore
            elif callable(o):
                meta_data[self.key_prefix + "_custom_" + str(custom_index)] = _compute(o, img_np)
                custom_index += 1
            else:
                raise ValueError("ops must be key string for predefined operations or callable function.")

        return img, meta_data


class ToDevice(Transform):
    """
    Move PyTorch Tensor to the specified device.
    It can help cache data into GPU and execute following logic on GPU directly.

    Note:
        If moving data to GPU device in the multi-processing workers of DataLoader, may got below CUDA error:
        "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing,
        you must use the 'spawn' start method."
        So usually suggest to set `num_workers=0` in the `DataLoader` or `ThreadDataLoader`.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, device: Union[torch.device, str], **kwargs) -> None:
        """
        Args:
            device: target device to move the Tensor, for example: "cuda:1".
            kwargs: other args for the PyTorch `Tensor.to()` API, for more details:
                https://pytorch.org/docs/stable/generated/torch.Tensor.to.html.

        """
        self.device = device
        self.kwargs = kwargs

    def __call__(self, img: torch.Tensor):
        if not isinstance(img, torch.Tensor):
            raise ValueError("img must be PyTorch Tensor, consider converting img by `EnsureType` transform first.")

        return img.to(self.device, **self.kwargs)


class CuCIM(Transform):
    """
    Wrap a non-randomized cuCIM transform, defined based on the transform name and args.
    For randomized transforms (or randomly applying a transform) use :py:class:`monai.transforms.RandCuCIM`.

    Args:
        name: the transform name in CuCIM package
        args: parameters for the CuCIM transform
        kwargs: parameters for the CuCIM transform

    Note:
        CuCIM transform only work with CuPy arrays, so this transform expects input data to be `cupy.ndarray`.
        Users can call `ToCuPy` transform to convert a numpy array or torch tensor to cupy array.
    """

    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__()
        self.name = name
        self.transform, _ = optional_import("cucim.core.operations.expose.transform", name=name)
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data):
        """
        Args:
            data: a CuPy array (`cupy.ndarray`) for the cuCIM transform

        Returns:
            `cupy.ndarray`

        """
        return self.transform(data, *self.args, **self.kwargs)


class RandCuCIM(CuCIM, RandomizableTransform):
    """
    Wrap a randomized cuCIM transform, defined based on the transform name and args,
    or randomly apply a non-randomized transform.
    For deterministic non-randomized transforms use :py:class:`monai.transforms.CuCIM`.

    Args:
        name: the transform name in CuCIM package.
        apply_prob: the probability to apply the transform (default=1.0)
        args: parameters for the CuCIM transform.
        kwargs: parameters for the CuCIM transform.

    Note:
        - CuCIM transform only work with CuPy arrays, so this transform expects input data to be `cupy.ndarray`.
          Users can call `ToCuPy` transform to convert a numpy array or torch tensor to cupy array.
        - If the cuCIM transform is already randomized the `apply_prob` argument has nothing to do with
          the randomness of the underlying cuCIM transform. `apply_prob` defines if the transform (either randomized
          or non-randomized) being applied randomly, so it can apply non-randomized transforms randomly but be careful
          with setting `apply_prob` to anything than 1.0 when using along with cuCIM's randomized transforms.
        - If the random factor of the underlying cuCIM transform is not derived from `self.R`,
          the results may not be deterministic. See Also: :py:class:`monai.transforms.Randomizable`.
    """

    def __init__(self, name: str, apply_prob: float = 1.0, *args, **kwargs) -> None:
        CuCIM.__init__(self, name, *args, **kwargs)
        RandomizableTransform.__init__(self, prob=apply_prob)

    def __call__(self, data):
        """
        Args:
            data: a CuPy array (`cupy.ndarray`) for the cuCIM transform

        Returns:
            `cupy.ndarray`

        """
        self.randomize(data)
        if not self._do_transform:
            return data
        return super().__call__(data)


class AddCoordinateChannels(Transform):
    """
    Appends additional channels encoding coordinates of the input. Useful when e.g. training using patch-based sampling,
    to allow feeding of the patch's location into the network.

    This can be seen as a input-only version of CoordConv:

    Liu, R. et al. An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution, NeurIPS 2018.

    Args:
        spatial_dims: the spatial dimensions that are to have their coordinates encoded in a channel and
            appended to the input image. E.g., `(0, 1, 2)` represents `H, W, D` dims and append three channels
            to the input image, encoding the coordinates of the input's three spatial dimensions.

    .. deprecated:: 0.8.0
        ``spatial_channels`` is deprecated, use ``spatial_dims`` instead.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    @deprecated_arg(
        name="spatial_channels", new_name="spatial_dims", since="0.8", msg_suffix="please use `spatial_dims` instead."
    )
    def __init__(self, spatial_dims: Sequence[int]) -> None:
        self.spatial_dims = spatial_dims

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            img: data to be transformed, assuming `img` is channel first.
        """
        if max(self.spatial_dims) > img.ndim - 2 or min(self.spatial_dims) < 0:
            raise ValueError(f"`spatial_dims` values must be within [0, {img.ndim - 2}]")

        spatial_size = img.shape[1:]
        coord_channels = np.array(np.meshgrid(*tuple(np.linspace(-0.5, 0.5, s) for s in spatial_size), indexing="ij"))
        coord_channels, *_ = convert_to_dst_type(coord_channels, img)  # type: ignore
        coord_channels = coord_channels[list(self.spatial_dims)]
        return concatenate((img, coord_channels), axis=0)
