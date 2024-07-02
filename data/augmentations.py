# Modified from P2PaLA

import argparse
import inspect
import pprint
import sys
from pathlib import Path
from typing import Optional, Sequence, override

import detectron2.data.transforms as T
import numpy as np
import torch
import torchvision.transforms.functional as F
from detectron2.config import CfgNode
from detectron2.data.transforms.augmentation import _get_aug_input_args
from scipy.ndimage import gaussian_filter

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))

from data import numpy_transforms as NT
from data import torch_transforms as TT

# REVIEW Use the self._init() function


class RandomApply(T.RandomApply):
    """
    Randomly apply an augmentation to an image with a given probability.
    """

    def __init__(self, tfm_or_aug: T.Augmentation | T.Transform, prob=0.5) -> None:
        """
        Randomly apply an augmentation to an image with a given probability.

        Args:
            tfm_or_aug (Augmentation | Transform): transform or augmentation to apply
            prob (float, optional): probability between 0.0 and 1.0 that
                the wrapper transformation is applied. Defaults to 0.5.
        """
        super().__init__(tfm_or_aug, prob)
        self.tfm_or_aug = self.aug

    def __repr__(self):
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                    param.kind != param.VAR_POSITIONAL and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(
                    self, name
                ), "Attribute {} not found! " "Default __repr__ only works if attributes match the constructor.".format(name)
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append("{}={}".format(name, attr_str))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()

    __str__ = __repr__


class Augmentation(T.Augmentation):
    def get_transform_aug_input(self, aug_input: T.AugInput) -> T.Transform:
        """
        Get the transform from the input

        Args:
            aug_input (T.AugInput): input to the augmentation

        Returns:
            T.Transform: transform
        """
        args = _get_aug_input_args(self, aug_input)
        transform = self.get_transform(*args)
        return transform

    def get_output_shape(self, old_height: int, old_width: int, dpi: Optional[int] = None) -> tuple[int, int]:
        """
        Get the output shape of the image

        Args:
            old_height (int): height of the image
            old_width (int): width of the image
            dpi (Optional[int], optional): dpi of the image. Defaults to None.

        Returns:
            tuple[int, int]: The output height and width of the image after applying the augmentation.
        """
        return (old_height, old_width)


class ResizeScaling(Augmentation):
    def __init__(self, scale: float, max_size: Optional[int] = None, target_dpi: Optional[int] = None) -> None:
        """
        Resize the image by a given scale

        Args:
            scale (float): scale percentage
            max_size (Optional[int], optional): max size of the image. Defaults to None.
            target_dpi (Optional[int], optional): target dpi of the image. Defaults to None.
        """
        super().__init__()
        self.scale = scale
        self.max_size = max_size
        self.target_dpi = target_dpi
        assert 0 < self.scale <= 1, "Scale percentage must be in range (0,1]"

    @override
    def get_output_shape(self, old_height: int, old_width: int, dpi: Optional[int] = None) -> tuple[int, int]:
        """
        Calculates the output shape of the image after applying the augmentation.

        Args:
            old_height (int): The original height of the image.
            old_width (int): The original width of the image.
            dpi (Optional[int]): The dots per inch of the image. Defaults to None.

        Returns:
            tuple[int, int]: The output height and width of the image after applying the augmentation.
        """
        scale = self.scale
        if self.target_dpi is not None and dpi is not None:
            scale = scale * self.target_dpi / dpi
        height, width = scale * old_height, scale * old_width

        # If max size is 0 or smaller assume no maxsize
        if self.max_size is None or self.max_size <= 0:
            max_size = sys.maxsize
        else:
            max_size = self.max_size
        if max(height, width) > max_size:
            scale = max_size * 1.0 / max(height, width)
            height = height * scale
            width = width * scale

        height = int(height + 0.5)
        width = int(width + 0.5)
        return (height, width)

    def numpy_transform(self, image: np.ndarray, dpi: Optional[int] = None) -> T.Transform:
        old_height, old_width, channels = image.shape
        height, width = self.get_output_shape(old_height, old_width, dpi=dpi)
        if (old_height, old_width) == (height, width):
            return T.NoOpTransform()

        return NT.ResizeTransform(old_height, old_width, height, width)

    def torch_transform(self, image: torch.Tensor, dpi: Optional[int] = None) -> T.Transform:
        old_height, old_width = image.shape[-2:]
        height, width = self.get_output_shape(old_height, old_width, dpi=dpi)

        if (old_height, old_width) == (height, width):
            return T.NoOpTransform()

        return TT.ResizeTransform(old_height, old_width, height, width)

    def get_transform(self, image: np.ndarray | torch.Tensor, dpi: Optional[int]) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image, dpi)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image, dpi)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class ResizeEdge(Augmentation):
    def __init__(
        self,
        min_size: int | Sequence[int],
        max_size: Optional[int] = None,
        sample_style: str = "choice",
        edge_length: Optional[int] = None,
    ) -> None:
        """
        Resize image alternative using cv2 instead of PIL or Pytorch

        Args:
            min_size (int | Sequence[int]): The minimum length of the side.
            max_size (int, optional): The maximum length of the other side. Defaults to None.
            sample_style (str, optional): The type of sampling used to get the output shape.
                Can be either "range" or "choice". Defaults to "choice".
            edge_length (int, optional): The edge length to be used if min_size is not a single value.
                Defaults to None.
        """
        super().__init__()
        assert sample_style in ["range", "choice"], sample_style
        if isinstance(min_size, int):
            min_size = (min_size, min_size)
        if sample_style == "range":
            assert len(min_size) == 2, "edge_length must be two values using 'range' sample style." f" Got {min_size}!"
        self.sample_style = sample_style
        self.min_size = min_size
        self.max_size = max_size
        self.edge_length = edge_length
        if len(set(min_size)) == 1:
            self.edge_length = min_size[0]

    @override
    def get_output_shape(self, old_height: int, old_width: int, dpi: Optional[int] = None) -> tuple[int, int]:
        """
        Calculates the output shape of the image after applying the augmentation.

        Args:
            old_height (int): The height of the original image.
            old_width (int): The width of the original image.
            dpi (Optional[int]): The DPI (dots per inch) of the image. Defaults to None.

        Returns:
            tuple[int, int]: The output shape of the image after applying the augmentation.

        Raises:
            ValueError: If the edge length is not set.
            NotImplementedError: If the method is not implemented in the subclass.
        """
        if self.edge_length is None:
            raise ValueError("Edge length is not set")
        # If edge length is 0 or smaller assume no resize
        if self.edge_length <= 0:
            return (old_height, old_width)
        raise NotImplementedError("This method should be implemented in the subclass")

    def numpy_transform(self, image: np.ndarray, dpi: Optional[int] = None) -> T.Transform:
        old_height, old_width, channels = image.shape
        height, width = self.get_output_shape(old_height, old_width, dpi=dpi)
        if (old_height, old_width) == (height, width):
            return T.NoOpTransform()

        return NT.ResizeTransform(old_height, old_width, height, width)

    def torch_transform(self, image: torch.Tensor, dpi: Optional[int] = None) -> T.Transform:
        old_height, old_width = image.shape[-2:]
        height, width = self.get_output_shape(old_height, old_width, dpi=dpi)

        if (old_height, old_width) == (height, width):
            return T.NoOpTransform()

        return TT.ResizeTransform(old_height, old_width, height, width)

    def get_transform(self, image: np.ndarray) -> T.Transform:

        if self.sample_style == "range":
            self.edge_length = np.random.randint(self.min_size[0], self.min_size[1] + 1)
        elif self.sample_style == "choice":
            self.edge_length = np.random.choice(self.min_size)
        else:
            raise ValueError('Only "choice" and "range" are accepted values')

        # If edge length is 0 or smaller assume no resize
        if self.edge_length <= 0:
            return T.NoOpTransform()

        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class ResizeShortestEdge(ResizeEdge):
    @override
    def get_output_shape(self, old_height: int, old_width: int, dpi: Optional[int] = None) -> tuple[int, int]:
        """
        Calculates the output shape of an image after applying the augmentation.

        Args:
            old_height (int): The original height of the image.
            old_width (int): The original width of the image.
            dpi (Optional[int]): The dots per inch of the image. Defaults to None.

        Returns:
            tuple[int, int]: The output height and width of the image after applying the augmentation.
        """
        if self.edge_length is None:
            raise ValueError("Edge length is not set")
        scale = float(self.edge_length) / min(old_height, old_width)
        if old_height < old_width:
            height, width = self.edge_length, scale * old_width
        else:
            height, width = scale * old_height, self.edge_length

        # If max size is 0 or smaller assume no maxsize
        if self.max_size is None or self.max_size <= 0:
            max_size = sys.maxsize
        else:
            max_size = self.max_size
        if max(height, width) > max_size:
            scale = max_size * 1.0 / max(height, width)
            height = height * scale
            width = width * scale

        height = int(height + 0.5)
        width = int(width + 0.5)
        return (height, width)


class ResizeLongestEdge(ResizeShortestEdge):
    @override
    def get_output_shape(self, old_height: int, old_width: int, dpi: Optional[int] = None) -> tuple[int, int]:
        """
        Calculates the output shape of an image after applying the augmentation.

        Args:
            old_height (int): The original height of the image.
            old_width (int): The original width of the image.
            dpi (Optional[int]): The dots per inch of the image. Defaults to None.

        Returns:
            tuple[int, int]: The output height and width of the image after applying the augmentation.
        """

        if self.edge_length is None:
            raise ValueError("Edge length is not set")
        scale = float(self.edge_length) / max(old_height, old_width)
        if old_height < old_width:
            height, width = self.edge_length, scale * old_width
        else:
            height, width = scale * old_height, self.edge_length

        # If max size is 0 or smaller assume no maxsize
        if self.max_size is None or self.max_size <= 0:
            max_size = sys.maxsize
        else:
            max_size = self.max_size
        if max(height, width) > max_size:
            scale = max_size * 1.0 / max(height, width)
            height = height * scale
            width = width * scale

        height = int(height + 0.5)
        width = int(width + 0.5)
        return (height, width)


class Flip(Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, horizontal: bool = True, vertical: bool = False) -> None:
        """
        Flip the image, XOR for horizontal or vertical

        Args:
            horizontal (boolean): whether to apply horizontal flipping. Defaults to True.
            vertical (boolean): whether to apply vertical flipping. Defaults to False.
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horizontal and vertical. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horizontal or vertical has to be True!")
        self.horizontal = horizontal
        self.vertical = vertical

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        height, width = image.shape[:2]

        if self.horizontal:
            return NT.HFlipTransform(width)
        elif self.vertical:
            return NT.VFlipTransform(height)
        else:
            raise ValueError("At least one of horizontal or vertical has to be True!")

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        height, width = image.shape[-2:]

        if self.horizontal:
            return TT.HFlipTransform(width)
        elif self.vertical:
            return TT.VFlipTransform(height)
        else:
            raise ValueError("At least one of horizontal or vertical has to be True!")

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomElastic(Augmentation):
    """
    Apply a random elastic transformation to the image, made using random warpfield and gaussian filters
    """

    def __init__(
        self,
        alpha: float = 0.1,
        sigma: float = 0.01,
        ignore_value: int = 255,
    ) -> None:
        """
        Apply a random elastic transformation to the image, made using random warpfield and gaussian filters

        Args:
            alpha (int, optional): scale factor of the warpfield (sets max value). Defaults to 0.045.
            stdv (int, optional): strength of the gaussian filter. Defaults to 0.01.
            ignore_value (int, optional): value that will be ignored during training. Defaults to 255.
        """
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.ignore_value = ignore_value

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        height, width = image.shape[:2]

        min_length = min(height, width)
        warpfield = np.zeros((height, width, 2))
        dx = gaussian_filter(((np.random.rand(height, width) * 2) - 1), self.sigma * min_length, mode="constant", cval=0)
        dy = gaussian_filter(((np.random.rand(height, width) * 2) - 1), self.sigma * min_length, mode="constant", cval=0)
        warpfield[..., 0] = dx * min_length * self.alpha
        warpfield[..., 1] = dy * min_length * self.alpha

        return NT.WarpFieldTransform(warpfield, ignore_value=self.ignore_value)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        height, width = image.shape[-2:]

        min_length = min(height, width)
        warpfield = torch.zeros((2, height, width), device=image.device)
        truncate = 4

        total_sigma = self.sigma * min_length
        pad = round(truncate * total_sigma)
        kernel_size = 2 * pad + 1
        random_x = torch.rand((1, height + 2 * pad, width + 2 * pad), device=image.device) * 2 - 1
        random_y = torch.rand((1, height + 2 * pad, width + 2 * pad), device=image.device) * 2 - 1

        dx = F.gaussian_blur(
            random_x,
            kernel_size=[kernel_size, kernel_size],
            sigma=[total_sigma, total_sigma],
        )[..., pad:-pad, pad:-pad]
        dy = F.gaussian_blur(
            random_y,
            kernel_size=[kernel_size, kernel_size],
            sigma=[total_sigma, total_sigma],
        )[..., pad:-pad, pad:-pad]

        warpfield[0] = dx * min_length * self.alpha
        warpfield[1] = dy * min_length * self.alpha

        return TT.WarpFieldTransform(warpfield, ignore_value=self.ignore_value)

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


# IDEA Use super class for RandomAffine, RandomTranslation, RandomRotation, RandomShear, RandomScale
class RandomAffine(Augmentation):
    """
    Apply a random affine transformation to the image
    """

    def __init__(
        self,
        t_stdv: float = 0.02,
        r_kappa: float = 30,
        sh_kappa: float = 20,
        sc_stdv: float = 0.12,
        probabilities: Optional[Sequence[float]] = None,
        ignore_value: int = 255,
    ) -> None:
        """
        Apply a random affine transformation to the image

        Args:
            t_stdv (float, optional): standard deviation used for the translation. Defaults to 0.02.
            r_kappa (float, optional): kappa value used for sampling the rotation. Defaults to 30.
            sh_kappa (float, optional): kappa value used for sampling the shear.. Defaults to 20.
            sc_stdv (float, optional): standard deviation used for the scale. Defaults to 0.12.
            probabilities (Optional[Sequence[float]], optional): individual probabilities for each sub category of an affine transformation. When None is given default to all 1.0 Defaults to None.
            ignore_value (int, optional): value that will be ignored during training. Defaults to 255.
        """
        super().__init__()
        self.t_stdv = t_stdv
        self.r_kappa = r_kappa
        self.sh_kappa = sh_kappa
        self.sc_stdv = sc_stdv
        self.ignore_value = ignore_value

        if probabilities is not None:
            assert len(probabilities) == 4, f"{len(probabilities)}: {probabilities}"
            self.probabilities = probabilities
        else:
            self.probabilities = [1.0] * 4

    def get_random_matrix(self, height: int, width: int):
        center = np.eye(3)
        center[:2, 2:] = np.asarray([width, height])[:, None] / 2

        uncenter = np.eye(3)
        uncenter[:2, 2:] = -1 * np.asarray([width, height])[:, None] / 2

        matrix = np.eye(3)

        # Translation
        if self._rand_range() < self.probabilities[0]:
            matrix[0:2, 2] = ((np.random.rand(2) - 1) * 2) * np.asarray([width, height]) * self.t_stdv

        # Rotation
        if self._rand_range() < self.probabilities[1]:
            rot = np.eye(3)
            theta = np.random.vonmises(0.0, self.r_kappa)
            rot[0:2, 0:2] = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

            matrix = matrix @ center @ rot @ uncenter

        # Shear
        if self._rand_range() < self.probabilities[2]:
            theta1 = np.random.vonmises(0.0, self.sh_kappa)

            shear1 = np.eye(3)
            shear1[0, 1] = theta1

            matrix = matrix @ center @ shear1 @ uncenter

            theta2 = np.random.vonmises(0.0, self.sh_kappa)

            shear2 = np.eye(3)
            shear2[1, 0] = theta2

            matrix = matrix @ center @ shear2 @ uncenter

        # Scale
        if self._rand_range() < self.probabilities[3]:
            scale = np.eye(3)
            scale[0, 0], scale[1, 1] = np.exp(np.random.rand(2) * self.sc_stdv)

            matrix = matrix @ center @ scale @ uncenter

        return matrix

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        height, width = image.shape[:2]

        matrix = self.get_random_matrix(height, width)

        return NT.AffineTransform(matrix, height=height, width=width, ignore_value=self.ignore_value)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        height, width = image.shape[-2:]

        matrix = self.get_random_matrix(height, width)
        matrix = torch.from_numpy(matrix).to(image.device).to(dtype=torch.float32)

        return TT.AffineTransform(matrix, height=height, width=width, ignore_value=self.ignore_value)

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if not any(self.probabilities):
            return T.NoOpTransform()

        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomTranslation(Augmentation):
    """
    Apply a random translation to the image
    """

    def __init__(self, t_stdv: float = 0.02, ignore_value: int = 255) -> None:
        """
        Apply a random affine transformation to the image

        Args:
            t_stdv (float, optional): standard deviation used for the translation. Defaults to 0.02.
            ignore_value (int, optional): value that will be ignored during training. Defaults to 255.
        """
        super().__init__()
        self.t_stdv = t_stdv
        self.ignore_value = ignore_value

    def get_random_matrix(self, height: int, width: int):
        matrix = np.eye(3)

        # Translation
        matrix[0:2, 2] = ((np.random.rand(2) - 1) * 2) * np.asarray([width, height]) * self.t_stdv

        return matrix

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        height, width = image.shape[:2]

        matrix = self.get_random_matrix(height, width)

        return NT.AffineTransform(matrix, height=height, width=width, ignore_value=self.ignore_value)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        height, width = image.shape[-2:]

        matrix = self.get_random_matrix(height, width)

        matrix = torch.from_numpy(matrix).to(device=image.device, dtype=torch.float32)

        return TT.AffineTransform(matrix, height=height, width=width, ignore_value=self.ignore_value)

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomRotation(Augmentation):
    """
    Apply a random rotation to the image
    """

    def __init__(self, r_kappa: float = 30, ignore_value: int = 255) -> None:
        """
        Apply a random rotation to the image

        Args:
            r_kappa (float, optional): kappa value used for sampling the rotation. Defaults to 30.
            ignore_value (int, optional): value that will be ignored during training. Defaults to 255.
        """
        super().__init__()
        self.r_kappa = r_kappa
        self.ignore_value = ignore_value

    def get_random_matrix(self, height: int, width: int):
        center = np.eye(3)
        center[:2, 2:] = np.asarray([width, height])[:, None] / 2

        uncenter = np.eye(3)
        uncenter[:2, 2:] = -1 * np.asarray([width, height])[:, None] / 2

        # Rotation
        rot = np.eye(3)
        theta = np.random.vonmises(0.0, self.r_kappa)
        rot[0:2, 0:2] = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]

        matrix = center @ rot @ uncenter

        return matrix

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        height, width = image.shape[:2]

        matrix = self.get_random_matrix(height, width)

        return NT.AffineTransform(matrix, height=height, width=width, ignore_value=self.ignore_value)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        height, width = image.shape[-2:]

        matrix = self.get_random_matrix(height, width)
        matrix = torch.from_numpy(matrix).to(device=image.device, dtype=torch.float32)

        return TT.AffineTransform(matrix, height=height, width=width, ignore_value=self.ignore_value)

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomShear(Augmentation):
    """
    Apply a random shearing to the image
    """

    def __init__(self, sh_kappa: float = 20, ignore_value: int = 255) -> None:
        """
        Apply a random shearing to the image

        Args:
            sh_kappa (float, optional): kappa value used for sampling the shear. Defaults to 20.
            ignore_value (int, optional): value that will be ignored during training. Defaults to 255.
        """
        super().__init__()
        self.sh_kappa = sh_kappa
        self.ignore_value = ignore_value

    def get_random_matrix(self, height: int, width: int):
        center = np.eye(3)
        center[:2, 2:] = np.asarray([width, height])[:, None] / 2

        uncenter = np.eye(3)
        uncenter[:2, 2:] = -1 * np.asarray([width, height])[:, None] / 2

        matrix = np.eye(3)

        # Shear1
        theta1 = np.random.vonmises(0.0, self.sh_kappa)

        shear1 = np.eye(3)
        shear1[0, 1] = theta1

        matrix = matrix @ center @ shear1 @ uncenter

        # Shear2
        theta2 = np.random.vonmises(0.0, self.sh_kappa)

        shear2 = np.eye(3)
        shear2[1, 0] = theta2

        matrix = matrix @ center @ shear2 @ uncenter

        return matrix

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        h, w = image.shape[:2]

        matrix = self.get_random_matrix(h, w)

        return NT.AffineTransform(matrix, height=h, width=w, ignore_value=self.ignore_value)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        h, w = image.shape[-2:]

        matrix = self.get_random_matrix(h, w)
        matrix = torch.from_numpy(matrix).to(device=image.device, dtype=torch.float32)

        return TT.AffineTransform(matrix, height=h, width=w, ignore_value=self.ignore_value)

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomScale(Augmentation):
    """
    Apply a random scaling to the image
    """

    def __init__(self, sc_stdv: float = 0.12, ignore_value: int = 255) -> None:
        """
        Apply a random scaling to the image

        Args:
            sc_stdv (float, optional): standard deviation used for the scale. Defaults to 0.12.
            ignore_value (int, optional): value that will be ignored during training. Defaults to 255.
        """
        super().__init__()
        self.sc_stdv = sc_stdv
        self.ignore_value = ignore_value

    def get_random_matrix(self, height: int, width: int):
        center = np.eye(3)
        center[:2, 2:] = np.asarray([width, height])[:, None] / 2

        uncenter = np.eye(3)
        uncenter[:2, 2:] = -1 * np.asarray([width, height])[:, None] / 2

        # Scale
        scale = np.eye(3)
        scale[0, 0], scale[1, 1] = np.exp(np.random.rand(2) * self.sc_stdv)

        matrix = center @ scale @ uncenter

        return matrix

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        h, w = image.shape[:2]

        matrix = self.get_random_matrix(h, w)

        return NT.AffineTransform(matrix, height=h, width=w, ignore_value=self.ignore_value)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        h, w = image.shape[-2:]

        matrix = self.get_random_matrix(h, w)
        matrix = torch.from_numpy(matrix).to(device=image.device, dtype=torch.float32)

        return TT.AffineTransform(matrix, height=h, width=w, ignore_value=self.ignore_value)

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class Grayscale(Augmentation):
    """
    Randomly convert the image to grayscale
    """

    def __init__(self, image_format="RGB") -> None:
        """
        Randomly convert the image to grayscale

        Args:
            image_format (str, optional): Color formatting. Defaults to "RGB".
        """
        super().__init__()
        self.image_format = image_format

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return NT.GrayscaleTransform(image_format=self.image_format)
        elif isinstance(image, torch.Tensor):
            return TT.GrayscaleTransform(image_format=self.image_format)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class Invert(Augmentation):
    """
    Invert the image
    """

    def __init__(self, max_value=255) -> None:
        """
        Invert the image
        """
        super().__init__()
        self.max_value = max_value

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        return NT.BlendTransform(src_image=np.asarray(self.max_value), src_weight=1, dst_weight=-1)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        return TT.BlendTransform(src_image=torch.tensor(self.max_value, device=image.device), src_weight=1, dst_weight=-1)

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomJPEGCompression(Augmentation):
    """
    Apply JPEG compression to the image
    """

    def __init__(self, min_quality: int = 40, max_quality: int = 100) -> None:
        """
        Apply JPEG compression to the image

        Args:
            quality_range (tuple[int, int], optional): range of the quality of the image. Defaults to (40, 100).
        """
        super().__init__()
        assert 0 <= min_quality <= 100, "Min quality must be in range [0, 100]"
        assert 0 <= max_quality <= 100, "Max quality must be in range [0, 100]"
        assert min_quality <= max_quality, "Min quality must be less than or equal to max quality"

        self.min_quality = min_quality
        self.max_quality = max_quality

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        quality = np.random.randint(self.min_quality, self.max_quality + 1)
        return NT.JPEGCompressionTransform(quality=quality)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        quality = np.random.randint(self.min_quality, self.max_quality + 1)
        return TT.JPEGCompressionTransform(quality=quality)

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomGaussianFilter(Augmentation):
    """
    Apply random gaussian kernels
    """

    def __init__(self, min_sigma: float = 1, max_sigma: float = 3, order: int = 0, iterations: int = 1) -> None:
        """
        Apply random gaussian kernels

        Args:
            min_sigma (float, optional): min Gaussian deviation. Defaults to 1.
            max_sigma (float, optional): max Gaussian deviation. Defaults to 3.
            order (int, optional): order of the gaussian kernel. Defaults to 0.
            iterations (int, optional): number of times the gaussian kernel is applied. Defaults to 1.
        """
        super().__init__()
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.order = order
        self.iterations = iterations

    def get_transform(self, image: np.ndarray) -> T.Transform:
        sigma = np.random.uniform(self.min_sigma, self.max_sigma)
        if isinstance(image, np.ndarray):
            return NT.GaussianFilterTransform(sigma=sigma, order=self.order, iterations=self.iterations)
        elif isinstance(image, torch.Tensor):
            return TT.GaussianFilterTransform(sigma=sigma, order=self.order, iterations=self.iterations)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomNoise(Augmentation):
    """
    Apply random noise to the image
    """

    def __init__(self, min_noise_std: float = 10, max_noise_std: float = 32) -> None:
        """
        Apply random noise to the image

        Args:
            min_noise_std (float, optional): min noise standard deviation. Defaults to 10.
            max_noise_std (float, optional): max noise standard deviation. Defaults to 32.
        """
        super().__init__()
        self.max_noise_std = max_noise_std
        self.min_noise_std = min_noise_std

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        std = np.random.uniform(self.min_noise_std, self.max_noise_std)
        noise = np.random.normal(0, std, image.shape)
        return NT.BlendTransform(src_image=noise, src_weight=1, dst_weight=1)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        std = np.random.uniform(self.min_noise_std, self.max_noise_std)
        noise = torch.randn_like(image, device=image.device, dtype=torch.float32) * std
        return TT.BlendTransform(src_image=noise, src_weight=1, dst_weight=1)

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomSaturation(Augmentation):
    """
    Change the saturation of an image

    Saturation intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce saturation
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase saturation
    """

    def __init__(self, intensity_min: float = 0.5, intensity_max: float = 1.5, image_format="RGB") -> None:
        """
        Change the saturation of an image

        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()

        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.image_format = image_format

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        if self.image_format == "RGB":
            grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif self.image_format == "BGR":
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise NotImplementedError(f"Image format {self.image_format} not supported")

        w = np.random.uniform(self.intensity_min, self.intensity_max)

        return NT.BlendTransform(grayscale, src_weight=1 - w, dst_weight=w)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        if self.image_format == "BGR":
            image = image[[2, 1, 0], ...]
        grayscale = F.rgb_to_grayscale(image)

        w = np.random.uniform(self.intensity_min, self.intensity_max)

        return TT.BlendTransform(grayscale, src_weight=1 - w, dst_weight=w)

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomContrast(Augmentation):
    """
    Randomly transforms image contrast

    Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce contrast
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase contrast

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min: float = 0.5, intensity_max: float = 1.5):
        """
        Randomly transforms image contrast

        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()

        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return NT.BlendTransform(src_image=image.astype(np.float32).mean(), src_weight=1 - w, dst_weight=w)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return TT.BlendTransform(src_image=image.to(dtype=torch.float32).mean(), src_weight=1 - w, dst_weight=w)

    def get_transform(self, image):
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomBrightness(Augmentation):
    """
    Randomly transforms image brightness

    Brightness intensity is uniformly sampled in (intensity_min, intensity_max).
    - intensity < 1 will reduce brightness
    - intensity = 1 will preserve the input image
    - intensity > 1 will increase brightness

    See: https://pillow.readthedocs.io/en/3.0.x/reference/ImageEnhance.html
    """

    def __init__(self, intensity_min: float = 0.5, intensity_max: float = 1.5):
        """
        Randomly transforms image brightness.

        Args:
            intensity_min (float): Minimum augmentation
            intensity_max (float): Maximum augmentation
        """
        super().__init__()

        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return NT.BlendTransform(src_image=np.zeros(1).astype(np.float32), src_weight=1 - w, dst_weight=w)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        w = np.random.uniform(self.intensity_min, self.intensity_max)
        return TT.BlendTransform(
            src_image=torch.zeros(1).to(device=image.device, dtype=torch.float32), src_weight=1 - w, dst_weight=w
        )

    def get_transform(self, image):
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomHue(Augmentation):
    """
    Randomly transforms image hue
    """

    def __init__(self, hue_delta_min: float = -0.5, hue_delta_max: float = 0.5, image_format="RGB") -> None:
        """
        Initialize the RandomHue class.

        Args:
            hue_delta_min (float, optional): lowest hue change. Defaults to -0.5.
            hue_delta_max (float, optional): highest hue change. Defaults to 0.5.
            image_format (str, optional): Color formatting. Defaults to "RGB".
        """
        super().__init__()
        self.hue_delta_min = hue_delta_min
        self.hue_delta_max = hue_delta_max
        self.image_format = image_format

    def get_transform(self, image: np.ndarray) -> T.Transform:
        hue_delta = np.random.uniform(self.hue_delta_min, self.hue_delta_max)

        if isinstance(image, np.ndarray):
            return NT.HueTransform(hue_delta, self.image_format)
        elif isinstance(image, torch.Tensor):
            return TT.HueTransform(hue_delta, self.image_format)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class AdaptiveThresholding(Augmentation):
    """
    Apply Adaptive thresholding to the image
    """

    def __init__(self, image_format="RGB") -> None:
        """
        Apply Adaptive thresholding to the image

        Args:
            image_format (str, optional): Color formatting. Defaults to "RGB".
        """
        super().__init__()
        self.image_format = image_format

    def get_transform(self, image: np.ndarray) -> T.Transform:
        if isinstance(image, np.ndarray):
            return NT.AdaptiveThresholdTransform(self.image_format)
        elif isinstance(image, torch.Tensor):
            return TT.AdaptiveThresholdTransform(self.image_format)
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class RandomOrientation(Augmentation):
    """
    Apply a random orientation to the image
    """

    def __init__(self, orientation_percentages: Optional[list[float | int]] = None) -> None:
        """
        Initialize the RandomOrientation class.

        Args:
            orientation_percentages (Optional[list[float | int]]): A list of orientation percentages.
                If None, default values of [1.0, 1.0, 1.0, 1.0] will be used.
        """
        super().__init__()
        self.orientation_percentages = orientation_percentages
        if self.orientation_percentages is None:
            self.orientation_percentages = [1.0] * 4
        array_percentages = np.asarray(self.orientation_percentages)
        assert len(array_percentages) == 4, f"{len(array_percentages)}: {array_percentages}"
        normalized_percentages = array_percentages / np.sum(array_percentages)
        self.normalized_percentages = normalized_percentages

    def get_transform(self, image) -> T.Transform:
        times_90_degrees = np.random.choice(4, p=self.normalized_percentages)
        if times_90_degrees == 0:
            return T.NoOpTransform()
        if isinstance(image, np.ndarray):
            return NT.OrientationTransform(times_90_degrees, image.shape[0], image.shape[1])
        elif isinstance(image, torch.Tensor):
            return TT.OrientationTransform(times_90_degrees, image.shape[-2], image.shape[-1])
        else:
            raise ValueError(f"Image type {type(image)} not supported")


class FixedSizeCrop(T.Augmentation):
    """
    If `crop_size` is smaller than the input image size, then it uses a random crop of
    the crop size. If `crop_size` is larger than the input image size, then it pads
    the right and the bottom of the image to the crop size if `pad` is True, otherwise
    it returns the smaller image.
    """

    def __init__(
        self,
        crop_size: tuple[int, int],
        pad: bool = True,
        pad_value: float = 0,
        seg_pad_value: int = 255,
    ):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value to the image.
            seg_pad_value: the padding value to the segmentation mask.
        """
        super().__init__()
        self.crop_size = crop_size
        self.pad = pad
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value

    def _get_crop(self, image: np.ndarray) -> T.Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add random crop if the image is scaled up.
        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0))
        offset = np.round(offset).astype(int)

        if isinstance(image, np.ndarray):
            return NT.CropTransform(offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0])
        elif isinstance(image, torch.Tensor):
            return TT.CropTransform(offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0])
        else:
            raise ValueError(f"Image type {type(image)} not supported")

    def _get_pad(self, image: np.ndarray) -> T.Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        original_size = np.minimum(input_size, output_size)
        if isinstance(image, np.ndarray):
            return NT.PadTransform(
                0, 0, pad_size[1], pad_size[0], original_size[1], original_size[0], self.pad_value, self.seg_pad_value
            )
        elif isinstance(image, torch.Tensor):
            return TT.PadTransform(
                0, 0, pad_size[1], pad_size[0], original_size[1], original_size[0], self.pad_value, self.seg_pad_value
            )
        else:
            raise ValueError(f"Image type {type(image)} not supported")

    def get_transform(self, image: np.ndarray) -> T.TransformList:
        transforms = [self._get_crop(image)]
        if self.pad:
            transforms.append(self._get_pad(image))
        return T.TransformList(transforms)


class RandomCrop(T.Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.

        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self.crop_type = crop_type
        self.crop_size = crop_size

    def numpy_transform(self, image: np.ndarray) -> T.Transform:
        height, width = image.shape[:2]
        croph, cropw = self.get_crop_size((height, width))
        h0 = np.random.randint(height - croph + 1)
        w0 = np.random.randint(width - cropw + 1)
        return NT.CropTransform(w0, h0, cropw, croph)

    def torch_transform(self, image: torch.Tensor) -> T.Transform:
        height, width = image.shape[-2:]
        croph, cropw = self.get_crop_size((height, width))
        h0 = np.random.randint(height - croph + 1)
        w0 = np.random.randint(width - cropw + 1)
        return TT.CropTransform(w0, h0, cropw, croph)

    def get_transform(self, image) -> T.Transform:
        if isinstance(image, np.ndarray):
            return self.numpy_transform(image)
        elif isinstance(image, torch.Tensor):
            return self.torch_transform(image)
        else:
            raise ValueError(f"Image type {type(image)} not supported")

    def get_crop_size(self, image_size) -> tuple[int, int]:
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomCrop_CategoryAreaConstraint(T.Augmentation):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that no single category
    occupies a ratio of more than `single_category_max_area` in semantic segmentation ground
    truth, which can cause unstability in training. The function attempts to find such a valid
    cropping window for at most 10 times.
    """

    def __init__(
        self,
        crop_type: str,
        crop_size,
        single_category_max_area: float = 1.0,
        ignored_category: Optional[int] = None,
    ):
        """
        Args:
            crop_type, crop_size: same as in :class:`RandomCrop`
            single_category_max_area: the maximum allowed area ratio of a
                category. Set to 1.0 to disable
            ignored_category: allow this category in the semantic segmentation
                ground truth to exceed the area ratio. Usually set to the category
                that's ignored in training.
        """
        self.crop_aug = RandomCrop(crop_type, crop_size)
        self.crop_type = crop_type
        self.crop_size = crop_size
        self.single_category_max_area = single_category_max_area
        self.ignored_category = ignored_category

    def numpy_transform(self, image: np.ndarray, sem_seg: np.ndarray) -> T.Transform:
        # TODO: Implement the numpy_transform method
        pass

    def torch_transform(self, image: torch.Tensor, sem_seg: torch.Tensor) -> T.Transform:
        # TODO: Implement the torch_transform method
        pass

    def get_transform(self, image, sem_seg) -> T.Transform:
        if self.single_category_max_area >= 1.0:
            return self.crop_aug.get_transform(image)
        else:
            h, w = sem_seg.shape
            x0 = 0
            y0 = 0
            crop_size = (0, 0)
            for _ in range(10):
                crop_size = self.crop_aug.get_crop_size((h, w))
                y0 = np.random.randint(h - crop_size[0] + 1)
                x0 = np.random.randint(w - crop_size[1] + 1)
                sem_seg_temp = sem_seg[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                if self.ignored_category is not None:
                    cnt = cnt[labels != self.ignored_category]
                if len(cnt) > 1 and np.max(cnt) < np.sum(cnt) * self.single_category_max_area:
                    break

            if isinstance(image, np.ndarray):
                return NT.CropTransform(x0, y0, crop_size[1], crop_size[0])
            elif isinstance(image, torch.Tensor):
                return TT.CropTransform(x0, y0, crop_size[1], crop_size[0])
            else:
                raise ValueError(f"Image type {type(image)} not supported")


def build_augmentation(cfg: CfgNode, mode: str = "train") -> list[T.Augmentation]:
    """
    Function to generate all the augmentations used in the inference and training process

    Args:
        cfg (CfgNode): The configuration node containing the parameters for the augmentations.
        mode (str): flag if the augmentation are used for inference or training
            - Possible values are "preprocess", "train", "val", or "test".

    Returns:
        list[T.Augmentation | T.Transform]: list of augmentations to apply to an image

    Raises:
        NotImplementedError: If the mode is not one of "train", "val", or "test".
        NotImplementedError: If the resize mode specified in the configuration is not recognized.
    """
    assert mode in ["preprocess", "train", "val", "test"], f"Unknown mode: {mode}"
    augmentation: list[T.Augmentation] = []

    if mode == "preprocess":
        if cfg.PREPROCESS.RESIZE.RESIZE_MODE == "none":
            augmentation.append(ResizeScaling(scale=1.0, target_dpi=cfg.PREPROCESS.DPI.TARGET_DPI))
        elif cfg.PREPROCESS.RESIZE.RESIZE_MODE in ["shortest_edge", "longest_edge"]:
            min_size = cfg.PREPROCESS.RESIZE.MIN_SIZE
            max_size = cfg.PREPROCESS.RESIZE.MAX_SIZE
            sample_style = cfg.PREPROCESS.RESIZE.RESIZE_SAMPLING
            if cfg.PREPROCESS.RESIZE.RESIZE_MODE == "shortest_edge":
                augmentation.append(ResizeShortestEdge(min_size, max_size, sample_style))
            elif cfg.PREPROCESS.RESIZE.RESIZE_MODE == "longest_edge":
                augmentation.append(ResizeLongestEdge(min_size, max_size, sample_style))
        elif cfg.PREPROCESS.RESIZE.RESIZE_MODE == "scaling":
            scaling = cfg.PREPROCESS.RESIZE.SCALING
            max_size = cfg.PREPROCESS.RESIZE.MAX_SIZE
            target_dpi = cfg.PREPROCESS.DPI.TARGET_DPI
            augmentation.append(ResizeScaling(scaling, max_size, target_dpi=target_dpi))
        else:
            raise NotImplementedError(f"{cfg.PREPROCESS.RESIZE.RESIZE_MODE} is not a known resize mode")
    else:
        if cfg.INPUT.RESIZE_MODE == "none":
            augmentation.append(ResizeScaling(scale=1.0, target_dpi=cfg.INPUT.DPI.TARGET_DPI))
        elif cfg.INPUT.RESIZE_MODE in ["shortest_edge", "longest_edge"]:
            if mode == "train":
                min_size = cfg.INPUT.MIN_SIZE_TRAIN
                max_size = cfg.INPUT.MAX_SIZE_TRAIN
                sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
            elif mode == "val":
                min_size = cfg.INPUT.MIN_SIZE_TEST
                max_size = cfg.INPUT.MAX_SIZE_TEST
                sample_style = "choice"
            elif mode == "test":
                min_size = cfg.INPUT.MIN_SIZE_TEST
                max_size = cfg.INPUT.MAX_SIZE_TEST
                sample_style = "choice"
            else:
                raise NotImplementedError(f"Unknown mode: {mode}")
            if cfg.INPUT.RESIZE_MODE == "shortest_edge":
                augmentation.append(ResizeShortestEdge(min_size, max_size, sample_style))
            elif cfg.INPUT.RESIZE_MODE == "longest_edge":
                augmentation.append(ResizeLongestEdge(min_size, max_size, sample_style))
        elif cfg.INPUT.RESIZE_MODE == "scaling":
            if mode == "train":
                max_size = cfg.INPUT.MAX_SIZE_TRAIN
                scaling = cfg.INPUT.SCALING_TRAIN
                target_dpi = cfg.INPUT.DPI.TARGET_DPI_TRAIN
            elif mode == "val":
                max_size = cfg.INPUT.MAX_SIZE_TRAIN
                scaling = cfg.INPUT.SCALING_TRAIN
                target_dpi = cfg.INPUT.DPI.TARGET_DPI_TRAIN
            elif mode == "test":
                max_size = cfg.INPUT.MAX_SIZE_TEST
                scaling = cfg.INPUT.SCALING_TEST
                target_dpi = cfg.INPUT.DPI.TARGET_DPI_TEST
            else:
                raise NotImplementedError(f"Unknown mode: {mode}")
            augmentation.append(ResizeScaling(scaling, max_size, target_dpi=target_dpi))
        else:
            raise NotImplementedError(f"{cfg.INPUT.RESIZE_MODE} is not a known resize mode")

    if not mode == "train":
        return augmentation

    # Crop
    if cfg.INPUT.CROP.ENABLED:
        augmentation.append(
            RandomCrop_CategoryAreaConstraint(
                crop_type=cfg.INPUT.CROP.TYPE,
                crop_size=cfg.INPUT.CROP.SIZE,
                single_category_max_area=cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
            )
        )

    # Moving pixels
    augmentation.append(
        RandomApply(
            RandomAffine(
                t_stdv=cfg.INPUT.AFFINE.TRANSLATION.STANDARD_DEVIATION,
                r_kappa=cfg.INPUT.AFFINE.ROTATION.KAPPA,
                sh_kappa=cfg.INPUT.AFFINE.SHEAR.KAPPA,
                sc_stdv=cfg.INPUT.AFFINE.SCALE.STANDARD_DEVIATION,
                probabilities=(
                    cfg.INPUT.AFFINE.TRANSLATION.PROBABILITY,
                    cfg.INPUT.AFFINE.ROTATION.PROBABILITY,
                    cfg.INPUT.AFFINE.SHEAR.PROBABILITY,
                    cfg.INPUT.AFFINE.SCALE.PROBABILITY,
                ),
                ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            ),
            prob=cfg.INPUT.AFFINE.PROBABILITY,
        )
    )

    augmentation.append(
        RandomApply(
            RandomElastic(
                alpha=cfg.INPUT.ELASTIC_DEFORMATION.ALPHA,
                sigma=cfg.INPUT.ELASTIC_DEFORMATION.SIGMA,
                ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            ),
            prob=cfg.INPUT.ELASTIC_DEFORMATION.PROBABILITY,
        )
    )

    augmentation.append(
        RandomApply(
            RandomGaussianFilter(
                min_sigma=cfg.INPUT.GAUSSIAN_FILTER.MIN_SIGMA,
                max_sigma=cfg.INPUT.GAUSSIAN_FILTER.MAX_SIGMA,
            ),
            prob=cfg.INPUT.GAUSSIAN_FILTER.PROBABILITY,
        )
    )

    # Flips
    augmentation.append(
        RandomApply(
            Flip(
                horizontal=True,
                vertical=False,
            ),
            prob=cfg.INPUT.HORIZONTAL_FLIP.PROBABILITY,
        )
    )
    augmentation.append(
        RandomApply(
            Flip(
                horizontal=True,
                vertical=False,
            ),
            prob=cfg.INPUT.HORIZONTAL_FLIP.PROBABILITY,
        )
    )

    # Orientation
    augmentation.append(
        RandomApply(
            RandomOrientation(
                orientation_percentages=cfg.INPUT.ORIENTATION.PERCENTAGES,
            ),
            prob=cfg.INPUT.ORIENTATION.PROBABILITY,
        )
    )

    # Color augments

    augmentation.append(
        RandomApply(
            RandomBrightness(
                intensity_min=cfg.INPUT.BRIGHTNESS.MIN_INTENSITY,
                intensity_max=cfg.INPUT.BRIGHTNESS.MAX_INTENSITY,
            ),
            prob=cfg.INPUT.BRIGHTNESS.PROBABILITY,
        )
    )
    augmentation.append(
        RandomApply(
            RandomContrast(
                intensity_min=cfg.INPUT.CONTRAST.MIN_INTENSITY,
                intensity_max=cfg.INPUT.CONTRAST.MAX_INTENSITY,
            ),
            prob=cfg.INPUT.CONTRAST.PROBABILITY,
        )
    )
    augmentation.append(
        RandomApply(
            RandomSaturation(
                intensity_min=cfg.INPUT.SATURATION.MIN_INTENSITY,
                intensity_max=cfg.INPUT.SATURATION.MAX_INTENSITY,
            ),
            prob=cfg.INPUT.SATURATION.PROBABILITY,
        )
    )
    augmentation.append(
        RandomApply(
            RandomHue(
                hue_delta_min=cfg.INPUT.HUE.MIN_DELTA,
                hue_delta_max=cfg.INPUT.HUE.MAX_DELTA,
            ),
            prob=cfg.INPUT.HUE.PROBABILITY,
        )
    )

    augmentation.append(
        RandomApply(
            RandomNoise(
                min_noise_std=cfg.INPUT.NOISE.MIN_STD,
                max_noise_std=cfg.INPUT.NOISE.MAX_STD,
            ),
            prob=cfg.INPUT.NOISE.PROBABILITY,
        )
    )

    augmentation.append(
        RandomApply(
            RandomJPEGCompression(
                min_quality=cfg.INPUT.JPEG_COMPRESSION.MIN_QUALITY,
                max_quality=cfg.INPUT.JPEG_COMPRESSION.MAX_QUALITY,
            ),
            prob=cfg.INPUT.JPEG_COMPRESSION.PROBABILITY,
        )
    )

    augmentation.append(
        RandomApply(
            Grayscale(
                image_format=cfg.INPUT.FORMAT,
            ),
            prob=cfg.INPUT.GRAYSCALE.PROBABILITY,
        )
    )

    augmentation.append(
        RandomApply(
            AdaptiveThresholding(
                image_format=cfg.INPUT.FORMAT,
            ),
            prob=cfg.INPUT.ADAPTIVE_THRESHOLDING.PROBABILITY,
        )
    )

    augmentation.append(
        RandomApply(
            Invert(
                max_value=cfg.INPUT.INVERT.MAX_VALUE,
            ),
            prob=cfg.INPUT.INVERT.PROBABILITY,
        )
    )

    return augmentation


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Testing the image augmentation and transformations")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input file", required=True, type=str)

    tmp_args = parser.add_argument_group("tmp files")
    tmp_args.add_argument("--tmp_dir", help="Temp files folder", type=str, default=None)
    tmp_args.add_argument("--keep_tmp_dir", action="store_true", help="Don't remove tmp dir after execution")

    detectron2_args = parser.add_argument_group("detectron2")
    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs="+", action="extend", help="optional args to change", default=[])
    args = parser.parse_args()

    return args


def test(args) -> None:
    from pathlib import Path

    import cv2
    from PIL import Image

    from core.setup import setup_cfg
    from data import preprocess
    from data.mapper import AugInput
    from utils.image_torch_utils import load_image_tensor_from_path
    from utils.image_utils import load_image_array_from_path
    from utils.tempdir import OptionalTemporaryDirectory

    input_path = Path(args.input)

    if not input_path.is_file():
        raise FileNotFoundError(f"Image {input_path} not found")

    cfg = setup_cfg(args)
    with OptionalTemporaryDirectory(name=args.tmp_dir, cleanup=not (args.keep_tmp_dir)) as tmp_dir:
        preprocesser = preprocess.Preprocess(cfg)
        preprocesser.set_output_dir(tmp_dir)
        output = preprocesser.process_single_file(input_path)

        # image = load_image_array_from_path(Path(tmp_dir).joinpath(output["image_paths"]))["image"]  # type: ignore
        # sem_seg = load_image_array_from_path(Path(tmp_dir).joinpath(output["sem_seg_paths"]), mode="grayscale")["image"]  # type: ignore

        image = load_image_tensor_from_path(Path(tmp_dir).joinpath(output["image_paths"]), device="cuda")["image"]  # type: ignore
        sem_seg = load_image_tensor_from_path(Path(tmp_dir).joinpath(output["sem_seg_paths"]), mode="grayscale", device="cuda")["image"]  # type: ignore

    # augs = build_augmentation(cfg, mode="train")
    # aug = T.AugmentationList(augs)

    augs = [RandomElastic()]
    aug = T.AugmentationList(augs)

    input_image = image.copy() if isinstance(image, np.ndarray) else image.clone()
    output = AugInput(image=input_image, sem_seg=sem_seg)
    transforms = aug(output)
    transforms = [t for t in transforms.transforms if not isinstance(t, T.NoOpTransform)]

    print(transforms)
    print(image.shape)
    print(image.dtype)
    print(image.min(), image.max())

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    im = Image.fromarray(image)
    im.show("Original")

    if isinstance(output.image, torch.Tensor):
        output.image = output.image.permute(1, 2, 0).cpu().numpy()

    im = Image.fromarray(output.image.round().clip(0, 255).astype(np.uint8))
    im.show("Transformed")

    if isinstance(output.sem_seg, torch.Tensor):
        output.sem_seg = output.sem_seg.permute(1, 2, 0).squeeze(-1).cpu().numpy()

    im = Image.fromarray(output.sem_seg.round().clip(0, 255).astype(np.uint8))
    im.show("Sem_Seg")


if __name__ == "__main__":
    args = get_arguments()
    test(args)
