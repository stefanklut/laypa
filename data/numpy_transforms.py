import argparse
import time
from typing import Optional

import cv2
import detectron2.data.transforms as T
import numpy as np
import shapely.geometry as geometry
from scipy.ndimage import gaussian_filter, map_coordinates


class TimedTransform(T.Transform):
    """
    A wrapper class to time the transforms.

    Args:
        T (T.Transform): Transform class to be timed
    """

    def __init_subclass__(cls) -> None:
        """
        Initialize the subclass. Apply the timing to the class methods.
        """
        super().__init_subclass__()
        cls.apply_image = cls._timed(cls.apply_image)
        cls.apply_segmentation = cls._timed(cls.apply_segmentation)
        cls.apply_coords = cls._timed(cls.apply_coords)
        cls.inverse = cls._timed(cls.inverse)

    @classmethod
    def _timed(cls, func):
        """
        Decorator to time the function

        Args:
            func (function): function to be timed
        """

        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            print(f"{cls.__name__}:{func.__name__} took {time.perf_counter() - start} seconds")
            return result

        return wrapper


# T.Transform = TimedTransform


class ResizeTransform(T.Transform):
    """
    Resize image using cv2
    """

    def __init__(self, height: int, width: int, new_height: int, new_width: int) -> None:
        """
        Resize image using cv2

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

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize Image

        Args:
            img (np.ndarray): image array HxWxC

        Returns:
            np.ndarray: resized images
        """
        img = img.astype(np.uint8)
        old_height, old_width, channels = img.shape
        assert (old_height, old_width) == (self.height, self.width), "Input dims do not match specified dims"

        resized_image = cv2.resize(img, (self.new_width, self.new_height), interpolation=cv2.INTER_LINEAR)

        return resized_image

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Resize coords

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            np.ndarray: resized coordinates
        """
        coords[:, 0] = coords[:, 0] * (self.new_width * 1.0 / self.width)
        coords[:, 1] = coords[:, 1] * (self.new_height * 1.0 / self.height)
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Resize segmentation (using nearest neighbor interpolation)

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: resized segmentation
        """
        old_height, old_width = segmentation.shape
        assert (old_height, old_width) == (self.height, self.width), "Input dims do not match specified dims"

        resized_segmentation = cv2.resize(segmentation, (self.new_width, self.new_height), interpolation=cv2.INTER_NEAREST)

        return resized_segmentation

    def inverse(self) -> T.Transform:
        """
        Inverse the resize by flipping old and new height
        """
        return ResizeTransform(self.new_height, self.new_width, self.height, self.width)


class HFlipTransform(T.Transform):
    """
    Perform horizontal flip. Taken from fvcore
    """

    def __init__(self, width: int):
        """
        Perform horizontal flip

        Args:
            width (int): image width
        """
        super().__init__()
        self.width = width

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (np.ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            np.ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        img = img.astype(np.uint8)
        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=1)
        else:
            return np.flip(img, axis=-2)

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

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (np.ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            np.ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        img = img.astype(np.uint8)
        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=0)
        else:
            return np.flip(img, axis=-3)

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
    """
    Apply a warp field (optical flow) to an image
    """

    def __init__(self, warpfield: np.ndarray, ignore_value=255) -> None:
        """
        Apply a warp field (optical flow) to an image

        Args:
            warpfield (np.ndarray): flow of pixels in the image
            ignore_value (int, optional): value to ignore in the segmentation. Defaults to 255.
        """
        super().__init__()
        self.warpfield = warpfield
        self.ignore_value = ignore_value

    @staticmethod
    def generate_grid(img: np.ndarray, warpfield: np.ndarray) -> np.ndarray:
        """
        Generate the new locations of pixels based on the offset warpfield

        Args:
            img (np.ndarray): of shape HxW or HxWxC
            warpfield (np.ndarray): HxW warpfield with movement per pixel

        Raises             :
        NotImplementedError: Only support for HxW and HxWxC right now

        Returns:
            np.ndarray: new pixel coordinates
        """

        if img.ndim == 2:
            height, width = img.shape
            x, y = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            indices = np.reshape(x + warpfield[..., 0], (-1, 1)), np.reshape(y + warpfield[..., 1], (-1, 1))
            return np.asarray(indices)
        elif img.ndim == 3:
            height, width, channels = img.shape
            x, y, z = np.meshgrid(np.arange(height), np.arange(width), np.arange(channels), indexing="ij")
            indices = (
                np.reshape(x + warpfield[..., 0, None], (-1, 1)),
                np.reshape(y + warpfield[..., 1, None], (-1, 1)),
                np.reshape(z, (-1, 1)),
            )
            return np.asarray(indices)
        else:
            raise NotImplementedError("No support for multi dimensions (NxHxWxC) right now")

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Warp an image with a specified warpfield, using spline interpolation

        Args:
            img (np.ndarray): image array HxWxC

        Returns:
            np.ndarray: warped image
        """
        img = img.astype(np.uint8)
        indices = self.generate_grid(img, self.warpfield)
        sampled_img = map_coordinates(img, indices, order=1, mode="constant", cval=0).reshape(img.shape)

        return sampled_img

    def apply_coords(self, coords: np.ndarray):
        """
        Coords moving might be possible but might move some out of bounds
        """
        # TODO This may be possible, and seems necessary for the instance predictions
        # raise NotImplementedError
        # IDEA self.recompute_boxes in dataset_mapper, with moving polygon values
        # HACK Currently just returning original coordinates
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Warp a segmentation with a specified warpfield, using spline interpolation with order 0

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: warped segmentation
        """
        indices = self.generate_grid(segmentation, self.warpfield)
        # cval=0 means background cval=self.ignore_value means ignored
        sampled_segmentation = map_coordinates(
            segmentation,
            indices,
            order=0,
            mode="constant",
            cval=self.ignore_value,
        ).reshape(segmentation.shape)

        return sampled_segmentation

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
        matrix: np.ndarray,
        height: int,
        width: int,
        ignore_value=255,
    ) -> None:
        """
        Apply an affine transformation to an image

        Args:
            matrix (np.ndarray): affine matrix applied to the pixels in image
            ignore_value (int, optional): value to ignore in the segmentation. Defaults to 255.
        """
        super().__init__()
        self.matrix = matrix.astype(np.float32)
        self.height = height
        self.width = width
        self.ignore_value = ignore_value

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply an affine transformation to the image

        Args:
            img (np.ndarray): image array HxWxC

        Raises:
            NotImplementedError: wrong dimensions of image

        Returns:
            np.ndarray: transformed image
        """
        img = img.astype(np.uint8)
        border_value = [0] if len(img.shape) == 2 else [0] * img.shape[-1]
        return cv2.warpAffine(
            img,
            self.matrix[:2, :],
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderValue=border_value,
        )

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply affine transformation to coordinates

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            np.ndarray: transformed coordinates
        """
        coords = coords.astype(np.float32)
        return cv2.transform(coords[:, None, :], self.matrix)[:, 0, :2]

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply an affine transformation to the segmentation

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: transformed segmentation
        """
        # borderValue=0 means background, borderValue=255 means ignored
        border_value = [255] if len(segmentation.shape) == 2 else [self.ignore_value] * min(segmentation.shape[-1], 4)
        return cv2.warpAffine(
            segmentation,
            self.matrix[:2, :],
            (segmentation.shape[1], segmentation.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=border_value,
        )

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

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Turn to grayscale by applying weights to the color image and than tile to get 3 channels again

        Args:
            img (np.ndarray): image array HxWxC

        Returns:
            np.ndarray: grayscale version of image
        """
        img = img.astype(np.uint8)
        if self.image_format == "BGR":
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        output = np.tile(grayscale[..., None], (1, 1, 3))
        return output

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Color transform does not affect coords

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Color transform does not affect segmentation

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: original segmentation
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

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply gaussian filters to the original image

        Args:
            img (np.ndarray): image array HxWxC

        Returns:
            np.ndarray: blurred image
        """
        img = img.astype(np.float32)
        transformed_img = img.copy()
        for _ in range(self.iterations):
            for i in range(img.shape[-1]):  # HxWxC
                transformed_img[..., i] = gaussian_filter(transformed_img[..., i], sigma=self.sigma, order=self.order)
        return transformed_img.astype(np.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Blurring should not affect the coordinates

        Args:
            coords (np.ndarray): loating point array of shape Nx2. Each row is
                (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Blurring should not affect the segmentation

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: original segmentation
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

    def __init__(self, src_image: np.ndarray, src_weight: float, dst_weight: float):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``

        Args:
            src_image (np.ndarray): Input image is blended with this image.
                The two images must have the same shape, range, channel order
                and dtype.
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
        """
        super().__init__()
        self.src_image = src_image
        self.src_weight = src_weight
        self.dst_weight = dst_weight

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply blend transform on the image(s).

        Args:
            img (np.ndarray): of shape NxHxWxC, or HxWxC or HxW. Assume the array is in range [0, 255].
        Returns:
            np.ndarray: blended image(s).
        """
        img = img.astype(np.float32)
        return np.clip(self.src_weight * self.src_image + self.dst_weight * img, 0, 255).astype(np.uint8)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: original segmentation
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
        self.hue_delta = hue_delta * 180
        self.color_space = color_space

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply hue transform to the image(s).

        Args:
            img (np.ndarray): image array assume the array is in range [0, 255].
        Returns:
            np.ndarray: hue transformed image(s).
        """
        img = img.astype(np.uint8)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img[:, :, 0] = (img[:, :, 0] + self.hue_delta) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (img[:, :, 0] + self.hue_delta) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
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

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: original segmentation
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

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Adaptive thresholding to the image.

        Args:
            img (np.ndarray): image array assume the array is in range [0, 255].
        Returns:
            np.ndarray: Adaptive thresholded image(s).
        """
        img = img.astype(np.uint8)

        # Convert to grayscale
        if self.image_format == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        thresholded = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        output = np.tile(thresholded[..., None], (1, 1, 3))
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

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: original segmentation
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

    @staticmethod
    def rotate(image: np.ndarray, times_90_degrees: int) -> np.ndarray:
        """
        Rotate the image by 90 degrees.

        Args:
            image (np.ndarray): Input image.
            times_90_degrees (int): Number of 90-degree rotations to apply. Should be between 0 and 3.

        Returns:
            np.ndarray: Rotated image.
        """
        if times_90_degrees == 0:
            return image
        if times_90_degrees == 1:
            if image.ndim == 2:
                return np.flip(np.transpose(image), axis=1)
            elif image.ndim == 3:
                return np.flip(np.transpose(image, (1, 0, 2)), axis=1)
            else:
                raise ValueError("Image should be either 2D or 3D")
        elif times_90_degrees == 2:
            return np.flip(np.flip(image, axis=0), axis=1)
        elif times_90_degrees == 3:
            if image.ndim == 2:
                return np.flip(np.transpose(image), axis=0)
            elif image.ndim == 3:
                return np.flip(np.transpose(image, (1, 0, 2)), axis=0)
            else:
                raise ValueError("Image should be either 2D or 3D")
        else:
            raise ValueError("Times 90 degrees should be between 0 and 3")

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply orientation change to the image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Rotated image.
        """
        rotated_img = self.rotate(img, self.times_90_degrees)
        return rotated_img

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
        Compress the image using JPEG compression. This is useful for simulating artifacts in low-quality images.

        Args:
            quality (int): JPEG compression quality. A number between 0 and 100.
        """
        super().__init__()
        self.quality = quality

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply JPEG compression to the image(s).

        Args:
            img (np.ndarray): image array assume the array is in range [0, 255].
        Returns:
            np.ndarray: JPEG compressed image(s).
        """
        img = img.astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        _, encoded_img = cv2.imencode(".jpg", img, encode_param)
        return cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.

        Args:
            coords (np.ndarray): floating point array of shape Nx2. Each row is (x, y).

        Returns:
            np.ndarray: original coords
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.

        Args:
            segmentation (np.ndarray): labels of shape HxW

        Returns:
            np.ndarray: original segmentation
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
        Crop the image(s) as a transform.

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

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).

        Args:
            img (np.ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            np.ndarray: cropped image(s).
        """
        if len(img.shape) <= 3:
            return img[self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w]
        else:
            return img[..., self.y0 : self.y0 + self.h, self.x0 : self.x0 + self.w, :]

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
            if isinstance(cropped, geometry.base.BaseMultipartGeometry):
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
        Pad the images as a transform.

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

    def apply_image(self, img):
        if img.ndim == 3:
            padding = ((self.y0, self.y1), (self.x0, self.x1), (0, 0))
        else:
            padding = ((self.y0, self.y1), (self.x0, self.x1))
        return np.pad(
            img,
            padding,
            mode="constant",
            constant_values=self.pad_value,
        )

    def apply_segmentation(self, img):
        if img.ndim == 3:
            padding = ((self.y0, self.y1), (self.x0, self.x1), (0, 0))
        else:
            padding = ((self.y0, self.y1), (self.x0, self.x1))
        return np.pad(
            img,
            padding,
            mode="constant",
            constant_values=self.seg_pad_value,
        )

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

    import cv2
    from PIL import Image

    input_path = Path(args.input)

    if not input_path.is_file():
        raise FileNotFoundError(f"Image {input_path} not found")

    print(f"Loading image {input_path}")
    image = cv2.imread(str(input_path))[..., ::-1]

    # output_image = OrientationTransform(1, image.shape[0], image.shape[1]).apply_image(image)
    # output_image = AdaptiveThresholdTransform().apply_image(image)
    output_image = AffineTransform(
        np.asarray(
            [
                [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
                [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                [0, 0, 1],
            ]
        ),
        image.shape[0],
        image.shape[1],
    ).apply_image(image)

    im = Image.fromarray(image)
    im.show("Original")

    im = Image.fromarray(output_image.round().clip(0, 255).astype(np.uint8))
    im.show("Transformed")


if __name__ == "__main__":
    args = get_arguments()
    test(args)
