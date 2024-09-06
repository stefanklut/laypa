import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import detectron2.data.transforms as T
import numpy as np
import shapely.geometry as geometry
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from data.numpy_transforms import TimedTransform

# T.Transform = TimedTransform


class ResizeTransform(T.Transform):
    """
    Resize image using torchvision
    """

    def __init__(self, height: int, width: int, new_height: int, new_width: int) -> None:
        """
        Resize image using torchvision

        Args:
            height (int): initial height
            width (int): initial width
            new_height (int): height after resizing
            new_width (int): width after resizing
        """
        super().__init__()
        self.height = height
        self.width = width
        self.new_height = new_height
        self.new_width = new_width

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Resize Image

        Args:
            img (torch.Tensor): image array CxHxW

        Returns:
            torch.Tensor: resized images
        """
        img = img.to(dtype=torch.uint8)
        channels, old_height, old_width = img.shape
        assert (old_height, old_width) == (
            self.height,
            self.width,
        ), f"Input dims ({old_height}, {old_width}) do not match specified dims ({self.height}, {self.width})"

        resized_image = F.resize(img, [self.new_height, self.new_width])

        return resized_image

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Resize coords

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            np.ndarray: resized coordinates
        """
        coords[:, 0] = coords[:, 0] * (self.new_width * 1.0 / self.width)
        coords[:, 1] = coords[:, 1] * (self.new_height * 1.0 / self.height)
        return coords

    def apply_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Resize segmentation (using nearest neighbor interpolation)

        Args:
            segmentation (torch.Tensor): labels of shape HxW

        Returns:
            torch.Tensor: resized segmentation
        """
        channels, old_height, old_width = segmentation.shape
        assert (old_height, old_width) == (
            self.height,
            self.width,
        ), f"Input dims ({old_height}, {old_width}) do not match specified dims ({self.height}, {self.width})"

        resized_segmentation = F.resize(
            segmentation, [self.new_height, self.new_width], interpolation=InterpolationMode.NEAREST
        )

        return resized_segmentation

    def inverse(self) -> T.Transform:
        """
        Inverse the resize by flipping old and new height
        """
        return ResizeTransform(self.new_height, self.new_width, self.height, self.width)


class HFlipTransform(T.Transform):
    """
    Perform horizontal flip
    """

    def __init__(self, width: int):
        """
        Perform horizontal flip

        Args:
            width (int): image width
        """
        super().__init__()
        self.width = width

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Flip the image(s).

        Args:
            img (torch.Tensor): of shape HxW, CxHxW or NxCxHxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            torch.Tensor: the flipped image(s).
        """
        img = img.to(dtype=torch.uint8)
        return F.hflip(img)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is (x, y).
        Returns:
            np.ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords

    def inverse(self) -> T.Transform:
        """
        The inverse is to flip again
        """
        return self


class VFlipTransform(T.Transform):
    """
    Perform vertical flip
    """

    def __init__(self, height: int):
        """
        Perform vertical flip

        Args:
            height (int): image height
        """
        super().__init__()
        self.height = height

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Flip the image(s).

        Args:
            img (torch.Tensor): of shape HxW, CxHxW or NxCxHxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            torch.Tensor: the flipped image(s).
        """
        img = img.to(dtype=torch.uint8)
        return F.vflip(img)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            np.ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 1] = self.height - coords[:, 1]
        return coords

    def inverse(self) -> T.Transform:
        """
        The inverse is to flip again
        """
        return self


class WarpFieldTransform(T.Transform):
    # TODO Convert this one
    """
    Apply a warp field (optical flow) to an image
    """

    def __init__(self, warpfield: torch.Tensor, ignore_value=255) -> None:
        """
        Apply a warp field (optical flow) to an image

        Args:
            warpfield (torch.Tensor): flow of pixels in the image
            ignore_value (int, optional): value to ignore in the segmentation. Defaults to 255.
        """
        super().__init__()
        self.indices = self.generate_grid(warpfield)
        self.ignore_value = ignore_value

    @staticmethod
    def generate_grid(warpfield: torch.Tensor) -> torch.Tensor:
        """
        Generate the new locations of pixels based on the offset warpfield

        Args:
            img (torch.Tensor): of shape HxW or CxHxW
            warpfield (torch.Tensor): 2xHxW warpfield with movement per pixel

        Raises             :
        NotImplementedError: Only support for HxW and CxHxW images

        Returns:
            torch.Tensor: new pixel coordinates
        """
        height = warpfield.shape[-2]
        width = warpfield.shape[-1]

        scale_warpfield = warpfield.clone()
        scale_warpfield[0] = scale_warpfield[0] * 2 / (height - 1)
        scale_warpfield[1] = scale_warpfield[1] * 2 / (width - 1)

        # Generate the new locations of pixels based on the offset warpfield
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        y = y.to(warpfield.device)
        x = x.to(warpfield.device)
        scale_y = y * 2 / (height - 1) - 1
        scale_x = x * 2 / (width - 1) - 1

        indices_y = scale_y + scale_warpfield[0]
        indices_x = scale_x + scale_warpfield[1]

        indices = torch.stack([indices_x, indices_y], dim=-1)

        return indices

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Warp an image with a specified warpfield, using spline interpolation

        Args:
            img (torch.Tensor): image array CxHxW

        Returns:
            torch.Tensor: warped image
        """
        img = img.to(dtype=torch.float32)
        sampled_img = torch.nn.functional.grid_sample(
            img[None, ...], self.indices[None, ...], mode="bilinear", padding_mode="zeros", align_corners=False
        )

        return sampled_img[0].to(dtype=torch.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Coords moving might be possible but might move some out of bounds
        """
        # TODO This may be possible, and seems necessary for the instance predictions
        # raise NotImplementedError
        # IDEA self.recompute_boxes in dataset_mapper, with moving polygon values
        # HACK Currently just returning original coordinates
        return coords

    def apply_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Warp a segmentation with a specified warpfield, using spline interpolation with order 0

        Args:
            segmentation (torch.Tensor): labels of shape HxW

        Returns:
            torch.Tensor: warped segmentation
        """
        segmentation = segmentation.to(dtype=torch.float32)[None, ...]
        segmentation = torch.concat([segmentation, torch.ones_like(segmentation)], dim=-3)

        sampled_segmentation = torch.nn.functional.grid_sample(
            segmentation, self.indices[None, ...], mode="nearest", padding_mode="zeros", align_corners=False
        )
        out_of_bounds = sampled_segmentation[..., -2:-1, :, :] == 0
        # Set out of bounds to ignore value (remove if you don't want to ignore)
        sampled_segmentation[..., 0:-1, :, :][out_of_bounds] = self.ignore_value

        sampled_segmentation = sampled_segmentation[..., 0:-1, :, :]

        return sampled_segmentation[0].to(dtype=torch.uint8)

    def inverse(self) -> T.Transform:
        """
        No inverse for a warp is possible since information is lost when moving out the visible window
        """
        raise NotImplementedError


class AffineTransform(T.Transform):
    """
    Apply an affine transformation to an image
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        height: int,
        width: int,
        ignore_value=255,
    ) -> None:
        """
        Apply an affine transformation to an image

        Args:
            matrix (torch.Tensor): affine matrix applied to the pixels in image
            height (int): height of the image
            width (int): width of the image
            ignore_value (int, optional): value to ignore in the segmentation. Defaults to 255.
        """
        super().__init__()
        self.numpy_matrix = matrix.cpu().numpy()
        self.torch_matrix = self.convert_matrix(matrix, height, width)
        self.height = height
        self.width = width

        self.ignore_value = ignore_value

    @staticmethod
    def convert_matrix(matrix: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Convert the matrix to the correct format for the affine transformation, between -1 and 1

        Args:
            matrix (torch.Tensor): affine matrix applied to the pixels in image
            height (int): height of the image
            width (int): width of the image

        Returns:
            torch.Tensor: converted matrix
        """
        param = torch.linalg.inv(matrix)
        converted_matrix = torch.zeros([3, 3])
        converted_matrix[0, 0] = param[0, 0]
        converted_matrix[0, 1] = param[0, 1] * height / width
        converted_matrix[0, 2] = param[0, 2] * 2 / width + converted_matrix[0, 0] + converted_matrix[0, 1] - 1
        converted_matrix[1, 0] = param[1, 0] * width / height
        converted_matrix[1, 1] = param[1, 1]
        converted_matrix[1, 2] = param[1, 2] * 2 / height + converted_matrix[1, 0] + converted_matrix[1, 1] - 1
        return converted_matrix

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply an affine transformation to the image

        Args:
            img (torch.Tensor): image array CxHxW

        Raises:
            NotImplementedError: wrong dimensions of image

        Returns:
            torch.Tensor: transformed image
        """
        img = img.to(dtype=torch.float32)

        affine_grid = torch.nn.functional.affine_grid(
            self.torch_matrix[None, :2], [1, 3, self.height, self.width], align_corners=False
        ).to(img.device)

        transformed_img = torch.nn.functional.grid_sample(
            img[None, ...], affine_grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return transformed_img[0].to(dtype=torch.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to coordinates

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            np.ndarray: transformed coordinates
        """
        coords = coords.astype(np.float32)
        return cv2.transform(coords[:, None, :], self.numpy_matrix)[:, 0, :2]

    def apply_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Apply an affine transformation to the segmentation

        Args:
            segmentation (torch.Tensor): labels of shape HxW

        Returns:
            torch.Tensor: transformed segmentation
        """
        segmentation = segmentation.to(dtype=torch.float32)[None, ...]

        # Add a channel to see what part of the image is out of bounds
        segmentation = torch.concat([segmentation, torch.ones_like(segmentation)], dim=-3)
        affine_grid = torch.nn.functional.affine_grid(
            self.torch_matrix[None, :2], [1, 2, self.height, self.width], align_corners=False
        ).to(segmentation.device)

        transformed_segmentation = torch.nn.functional.grid_sample(
            segmentation, affine_grid, mode="nearest", padding_mode="zeros", align_corners=False
        )

        out_of_bounds = transformed_segmentation[..., -2:-1, :, :] == 0

        # Set out of bounds to ignore value (remove if you don't want to ignore)
        transformed_segmentation[..., 0:-1, :, :][out_of_bounds] = self.ignore_value
        transformed_segmentation = transformed_segmentation[..., 0:-1, :, :]
        return transformed_segmentation[0].to(dtype=torch.uint8)

    def inverse(self) -> T.Transform:
        """
        Inverse not always possible, since information may be lost when moving out the visible window.
        """
        raise NotImplementedError


class GrayscaleTransform(T.Transform):
    """
    Convert an image to grayscale
    """

    def __init__(self, image_format: str = "RGB") -> None:
        """
        Convert an image to grayscale

        Args:
            image_format (str, optional): type of image format. Defaults to "RGB".
        """
        super().__init__()

        # Previously used to get the grayscale value
        # self.rgb_weights = np.asarray([0.299, 0.587, 0.114]).astype(np.float32)

        self.image_format = image_format

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Turn to grayscale by applying weights to the color image and than tile to get 3 channels again

        Args:
            img (torch.Tensor): image array CxHxW

        Returns:
            torch.Tensor: grayscale version of image
        """
        img = img.to(dtype=torch.uint8)
        if self.image_format == "BGR":
            img = img[[2, 1, 0], ...]

        img = F.rgb_to_grayscale(img, num_output_channels=3)
        return img

    def apply_coords(self, coords: torch.Tensor):
        """
        Color transform does not affect coords

        Args:
            coords (torch.Tensor): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            torch.Tensor: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Color transform does not affect segmentation

        Args:
            segmentation (torch.Tensor): labels of shape HxW

        Returns:
            torch.Tensor: original segmentation
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        No inverse possible. Grayscale cannot return to color
        """
        raise NotImplementedError


class GaussianFilterTransform(T.Transform):
    """
    Apply one or more gaussian filters
    """

    def __init__(self, sigma: float = 2, order: int = 0, iterations: int = 1) -> None:
        """
        Apply one or more gaussian filters

        Args:
            sigma (float, optional): Gaussian deviation. Defaults to 4.
            order (int, optional): order of gaussian derivative. Defaults to 0.
            iterations (int, optional): times the kernel is applied. Defaults to 1.
        """
        self.sigma = sigma
        self.order = order
        self.iterations = iterations

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply gaussian filters to the original image

        Args:
            img (torch.Tensor): image array CxHxW

        Returns:
            torch.Tensor: blurred image
        """
        img = img.to(dtype=torch.float32)
        truncate = 4

        kernel_size = 2 * round(truncate * self.sigma) + 1
        for _ in range(self.iterations):
            img = F.gaussian_blur(
                img,
                kernel_size=[kernel_size, kernel_size],
                sigma=[self.sigma, self.sigma],
            ).to(dtype=torch.uint8)

        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Blurring should not affect the coordinates

        Args:
            coords (np.ndarray): loating point array of shape Nx2. Each row is (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Blurring should not affect the segmentation

        Args:
            segmentation (torch.Tensor): labels of shape HxW

        Returns:
            torch.Tensor: original segmentation
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        No inverse of blurring is possible since information is lost
        """
        raise NotImplementedError


class BlendTransform(T.Transform):
    """
    Transforms pixel colors with PIL enhance functions.
    """

    def __init__(self, src_image: torch.Tensor, src_weight: float, dst_weight: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (torch.Tensor): Input image is blended with this image.
                The two images must have the same shape, range, channel order
                and dtype.
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self.src_image = src_image
        self.src_weight = src_weight
        self.dst_weight = dst_weight

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply blend transform on the image(s).

        Args:
            img (np.ndarray): of shape CxHxW. Assume the array is in range [0, 255].
        Returns:
            np.ndarray: blended image(s).
        """
        img = img.to(dtype=torch.float32)
        return torch.clip(self.src_weight * self.src_image + self.dst_weight * img, 0, 255).to(dtype=torch.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Apply no transform on the full-image segmentation.

        Args:
            segmentation (torch.Tensor): labels of shape HxW

        Returns:
            torch.Tensor: original segmentation
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        The inverse is not possible. The blend is not reversible.
        """
        raise NotImplementedError


class HueTransform(T.Transform):
    def __init__(self, hue_delta: float, color_space: str = "RGB"):
        """
        Args:
            delta (float): the amount to shift the hue channel. The hue channel is
                shifted in degrees within the range [-180, 180].
        """
        super().__init__()
        self.hue_delta = hue_delta
        self.color_space = color_space

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply hue transform to the image(s).

        Args:
            img (torch.Tensor): image array assume the array is in range [0, 255].
        Returns:
            np.ndarray: hue transformed image(s).
        """
        img = img.to(dtype=torch.uint8)
        if self.color_space == "BGR":
            img = img[[2, 1, 0], ...]
        img = F.adjust_hue(img, self.hue_delta)
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Apply no transform on the full-image segmentation.

        Args:
            segmentation (torch.Tensor): labels of shape HxW

        Returns:
            torch.Tensor: original segmentation
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        Compute the inverse of the transformation. The inverse of a hue change is a negative hue change.

        Returns:
            Transform: Inverse transformation.
        """
        return HueTransform(-self.hue_delta, self.color_space)


class AdaptiveThresholdTransform(T.Transform):
    def __init__(self, image_format: str = "RGB") -> None:
        """
        Apply Adaptive thresholding to an image.

        Args:
            image_format (str, optional): type of image format. Defaults to "RGB".
        """
        super().__init__()
        self.image_format = image_format

    @staticmethod
    def adaptive_threshold(image: torch.Tensor, kernel_size=3, C=0.1):
        """
        Apply adaptive thresholding to an image.

        Args:
            image (torch.Tensor): Grayscale image tensor of shape (1, H, W).
            kernel_size (int): Size of the neighborhood used to compute the threshold for each pixel.
            C (float): Constant subtracted from the mean or weighted mean.

        Returns:
            torch.Tensor: Thresholded image.
        """
        # Ensure the image tensor is float
        image = image.to(dtype=torch.float32)

        # Pad the image to handle borders
        pad_size = kernel_size // 2
        padded_image = F.pad(image, [pad_size, pad_size, pad_size, pad_size], padding_mode="reflect")

        # Compute the local mean
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=image.device) / (kernel_size**2)
        local_mean = torch.nn.functional.conv2d(padded_image, kernel, stride=1)

        # Apply the threshold
        thresholded = (image > local_mean - C).to(dtype=torch.uint8)
        thresholded = thresholded * 255

        return thresholded

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply Adaptive thresholding to the image.

        Args:
            img (torch.Tensor): image array assume the array is in range [0, 255].
        Returns:
            np.ndarray: Adaptive thresholded image(s).
        """
        img = img.to(dtype=torch.uint8)
        if self.image_format == "BGR":
            img = img[[2, 1, 0], ...]

        # Convert the image to grayscale
        gray_img = F.rgb_to_grayscale(img, num_output_channels=1)
        thresholded = self.adaptive_threshold(gray_img, kernel_size=11, C=2)
        output = thresholded.repeat(3, 1, 1)

        return output

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Apply no transform on the full-image segmentation.

        Args:
            segmentation (torch.Tensor): labels of shape HxW

        Returns:
            torch.Tensor: original segmentation
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        The inverse is not possible. Information is lost during Adaptive thresholding.
        """
        raise NotImplementedError


class OrientationTransform(T.Transform):
    """
    Transform that applies 90 degrees rotation to an image and its corresponding coordinates.
    """

    def __init__(self, times_90_degrees: int, height: int, width: int) -> None:
        """
        Transform that applies 90 degrees rotation to an image and its corresponding coordinates.

        Args:
            times_90_degrees (int): Number of 90-degree rotations to apply. Should be between 0 and 3.
            height (int): Height of the image.
            width (int): Width of the image.
        """
        super().__init__()
        self.times_90_degrees = times_90_degrees % 4
        self.height = height
        self.width = width

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply orientation change to the image.

        Args:
            img (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Rotated image.
        """
        return torch.rot90(img, -self.times_90_degrees, dims=[1, 2])

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply orientation change to the coordinates.

        Args:
            coords (np.ndarray): Input coordinates.

        Returns:
            np.ndarray: Rotated coordinates.
        """
        if self.times_90_degrees == 0:
            return coords
        elif self.times_90_degrees == 1:
            new_coords = coords.copy()
            new_coords[:, 0], new_coords[:, 1] = self.height - coords[:, 1], coords[:, 0]
            return new_coords
        elif self.times_90_degrees == 2:
            new_coords = coords.copy()
            new_coords[:, 0], new_coords[:, 1] = self.width - coords[:, 1], self.height - coords[:, 0]
            return new_coords
        elif self.times_90_degrees == 3:
            new_coords = coords.copy()
            new_coords[:, 0], new_coords[:, 1] = coords[:, 1], self.width - coords[:, 0]
            return new_coords
        else:
            raise ValueError("Times 90 degrees should be between 0 and 3")

    def inverse(self) -> T.Transform:
        """
        Compute the inverse of the transformation. The inverse of a 90-degree rotation is a 270-degree rotation to get the original image.

        Returns:
            Transform: Inverse transformation.
        """
        if self.times_90_degrees % 2 == 0:
            height, width = self.height, self.width
        else:
            width, height = self.width, self.height
        return OrientationTransform(4 - self.times_90_degrees, height, width)


class JPEGCompressionTransform(T.Transform):
    def __init__(self, quality: int):
        """
        Args:
            quality (int): JPEG compression quality. A number between 0 and 100.
        """
        super().__init__()
        self.quality = quality

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply JPEG compression to the image(s).

        Args:
            img (torch.Tensor): image array assume the array is in range [0, 255].
        Returns:
            torch.Tensor: JPEG compressed image(s).
        """
        img = img.to(dtype=torch.uint8)
        encoded = torchvision.io.encode_jpeg(img.cpu(), quality=self.quality)
        return torchvision.io.decode_jpeg(encoded, device=img.device)  # type: ignore

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: torch.Tensor) -> torch.Tensor:
        """
        Apply no transform on the full-image segmentation.

        Args:
            segmentation (torch.Tensor): labels of shape HxW

        Returns:
            torch.Tensor: original segmentation
        """
        return segmentation

    def inverse(self) -> T.Transform:
        """
        The inverse is not possible. Information is lost during JPEG compression.
        """
        raise NotImplementedError


class CropTransform(T.Transform):
    def __init__(
        self,
        x0: int,
        y0: int,
        w: int,
        h: int,
        orig_w: Optional[int] = None,
        orig_h: Optional[int] = None,
    ):
        """
        Args:
            x0, y0, w, h (int): crop the image(s) by img[y0:y0+h, x0:x0+w].
            orig_w, orig_h (int): optional, the original width and height
                before cropping. Needed to make this transform invertible.
        """
        super().__init__()
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.orig_w = orig_w
        self.orig_h = orig_h

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Crop the image(s).

        Args:
            img (torch.Tensor): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            torch.Tensor: cropped image(s).
        """
        return img[..., self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            np.ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.

        Args:
            polygon (list[np.ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            list[np.ndarray]: cropped polygons.
        """

        # Create a window that will be used to crop
        crop_box = geometry.box(self.x0, self.y0, self.x0 + self.w, self.y0 + self.h).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            if not polygon.is_valid:
                continue
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                cropped = cropped.geoms
            else:
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]

    def inverse(self) -> T.Transform:
        assert (
            self.orig_w is not None and self.orig_h is not None
        ), "orig_w, orig_h are required for CropTransform to be invertible!"
        pad_x1 = self.orig_w - self.x0 - self.w
        pad_y1 = self.orig_h - self.y0 - self.h
        return PadTransform(self.x0, self.y0, pad_x1, pad_y1, orig_w=self.w, orig_h=self.h)


class PadTransform(T.Transform):
    def __init__(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        orig_w: Optional[int] = None,
        orig_h: Optional[int] = None,
        pad_value: float = 0,
        seg_pad_value: int = 0,
    ):
        """
        Args:
            x0, y0: number of padded pixels on the left and top
            x1, y1: number of padded pixels on the right and bottom
            orig_w, orig_h: optional, original width and height.
                Needed to make this transform invertible.
            pad_value: the padding value to the image
            seg_pad_value: the padding value to the segmentation mask
        """
        super().__init__()
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.pad_value = pad_value
        self.seg_pad_value = seg_pad_value

    def apply_image(self, img: torch.Tensor) -> torch.Tensor:
        padding = [self.x0, self.y0, self.x1, self.y1]
        return F.pad(img, padding, fill=self.pad_value)

    def apply_segmentation(self, img):
        padding = [self.x0, self.y0, self.x1, self.y1]
        return F.pad(img, padding, fill=self.seg_pad_value)

    def apply_coords(self, coords):
        coords[:, 0] += self.x0
        coords[:, 1] += self.y0
        return coords

    def inverse(self) -> T.Transform:
        assert (
            self.orig_w is not None and self.orig_h is not None
        ), "orig_w, orig_h are required for PadTransform to be invertible!"
        neww = self.orig_w + self.x0 + self.x1
        newh = self.orig_h + self.y0 + self.y1
        return CropTransform(self.x0, self.y0, self.orig_w, self.orig_h, orig_w=neww, orig_h=newh)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Testing the image augmentation and transformations")
    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input file", required=True, type=str)

    args = parser.parse_args()
    return args


def test(args) -> None:
    from pathlib import Path

    from PIL import Image

    input_path = Path(args.input)

    if not input_path.is_file():
        raise FileNotFoundError(f"Image {input_path} not found")

    print(f"Loading image {input_path}")
    image = torchvision.io.read_image(str(input_path)).to("cuda")

    # output_image = ResizeTransform(image.shape[1], image.shape[2], 256, 256).apply_image(image)
    # output_image = HFlipTransform(image.shape[2]).apply_image(image)
    # output_image = VFlipTransform(image.shape[1]).apply_image(image)
    # output_image = WarpFieldTransform(torch.ones(2, image.shape[1], image.shape[2], device=image.device) * 20).apply_image(
    #     image
    # )
    output_image = AffineTransform(
        torch.Tensor(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        ),
        image.shape[1],
        image.shape[2],
    ).apply_segmentation(torch.zeros([1, 4000, 4000], dtype=torch.uint8))
    # output_image = GrayscaleTransform().apply_image(image)
    # output_image = GaussianFilterTransform(sigma=2).apply_image(image)
    # output_image = BlendTransform(1.0, 0.5, 0.5).apply_image(image)
    # output_image = HueTransform(0.5).apply_image(image)
    # output_image = AdaptiveThresholdTransform().apply_image(image)
    # output_image = OrientationTransform(1, image.shape[0], image.shape[1]).apply_image(image)

    output_image = output_image.cpu().numpy().squeeze(0)

    im = Image.fromarray(image.cpu().numpy().transpose(1, 2, 0))
    im.show("Original")

    im = Image.fromarray(output_image.round().clip(0, 255).astype(np.uint8))
    im.show("Transformed")


if __name__ == "__main__":
    args = get_arguments()
    test(args)
