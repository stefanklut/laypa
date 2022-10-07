# Taken from P2PaLA

import argparse
import numpy as np
import torch

import detectron2.data.transforms as T

from scipy.ndimage import map_coordinates
from scipy.ndimage import affine_transform
from scipy.ndimage import gaussian_filter

# TODO Check if there is a benefit for using scipy instead of the standard torchvision


class HFlipTransform(T.Transform):
    """
    Perform horizontal flip. Taken from fvcore
    """

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=1)
        else:
            return np.flip(img, axis=-2)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

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
    Perform vertical flip.
    """

    def __init__(self, height: int):
        super().__init__()
        self.height = height

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).

        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=0)
        else:
            return np.flip(img, axis=-3)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

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


class RandomFlip(T.Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, horizontal=True, vertical=False) -> None:
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self.prob = prob
        self.horizontal = horizontal
        self.vertical = vertical

    def get_transform(self, image) -> T.Transform:
        h, w = image.shape[:2]
        if self._rand_range() < self.prob:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        return T.NoOpTransform()


class WarpField(T.Transform):
    def __init__(self, warpfield: np.ndarray) -> None:
        """
        Args:
            warpfield (np.ndarray): flow of pixels in the image
        """
        super().__init__()
        self.warpfield = warpfield

    @staticmethod
    def generate_grid(img: np.ndarray, warpfield: np.ndarray):
        if img.ndim == 2:
            x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(
                img.shape[1]), indexing="ij")
            indices = np.reshape(
                x + warpfield[..., 0], (-1, 1)), np.reshape(y + warpfield[..., 1], (-1, 1))
            return np.asarray(indices)
        elif img.ndim == 3:
            x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(
                img.shape[1]), np.arange(img.shape[2]), indexing="ij")
            indices = np.reshape(x + warpfield[..., 0, None], (-1, 1)), np.reshape(
                y + warpfield[..., 1, None], (-1, 1)), np.reshape(z, (-1, 1))
            return np.asarray(indices)
        else:
            raise NotImplementedError(
                "No support for multi dimensions (NxHxWxC) right now")

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        indices = self.generate_grid(img, self.warpfield)
        sampled_img = map_coordinates(
            img, indices, order=1, mode="constant", cval=0).reshape(img.shape)

        return sampled_img

    def apply_coords(self, coords: np.ndarray):
        raise NotImplementedError

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        indices = self.generate_grid(segmentation, self.warpfield)
        sampled_segmentation = map_coordinates(
            segmentation, indices, order=0, mode="constant", cval=255).reshape(segmentation.shape)

        return sampled_segmentation

    def inverse(self) -> T.Transform:
        raise NotImplementedError


class RandomElastic(T.Augmentation):
    def __init__(self, prob=0.5, alpha=34, stdv=4) -> None:
        super().__init__()
        self.prob = prob
        self.alpha = alpha
        self.stdv = stdv

    def get_transform(self, image) -> T.Transform:
        if self._rand_range() < self.prob:
            h, w = image.shape[:2]
            warpfield = np.zeros((h, w, 2))
            dx = gaussian_filter(
                ((np.random.rand(h, w) * 2) - 1), self.stdv, mode="constant", cval=0)
            dy = gaussian_filter(
                ((np.random.rand(h, w) * 2) - 1), self.stdv, mode="constant", cval=0)
            warpfield[..., 0] = dx * self.alpha
            warpfield[..., 1] = dy * self.alpha

            return WarpField(warpfield)

        return T.NoOpTransform()


class AffineTransform(T.Transform):
    def __init__(self, matrix: np.ndarray) -> None:
        """
        Args:
            warpfield (np.ndarray): flow of pixels in the image
        """
        super().__init__()
        self.matrix = matrix

    def apply_image(self, img: np.ndarray) -> np.ndarray:

        if img.ndim == 2:
            return affine_transform(img, self.matrix, order=1, mode='constant', cval=0)
        elif img.ndim == 3:
            transformed_img = np.empty_like(img)
            for i in range(img.shape[-1]):  # HxWxC
                transformed_img[..., i] = affine_transform(
                    img[..., i], self.matrix, order=1, mode='constant', cval=0)
            return transformed_img
        else:
            raise NotImplementedError(
                "No support for multi dimensions (NxHxWxC) right now")

    def apply_coords(self, coords: np.ndarray):
        return super().apply_coords(coords)

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return affine_transform(segmentation, self.matrix, order=0, mode='constant', cval=255)

    def inverse(self) -> T.Transform:
        raise NotImplementedError


class RandomAffine(T.Augmentation):
    def __init__(self, prob=0.5, t_stdv=0.02, r_kappa=30, sh_kappa=20, sc_stdv=0.12) -> None:
        super().__init__()
        self.prob = prob
        self.t_stdv = t_stdv
        self.r_kappa = r_kappa
        self.sh_kappa = sh_kappa
        self.sc_stdv = sc_stdv

    def get_transform(self, image) -> T.Transform:
        if self._rand_range() < self.prob:
            h, w = image.shape[:2]

            center = np.eye(3)
            center[:2, 2:] = np.asarray([w, h])[:, None] / 2

            uncenter = np.eye(3)
            uncenter[:2, 2:] = -1 * np.asarray([w, h])[:, None] / 2

            matrix = np.eye(3)

            # Translation
            matrix[0:2, 2] = ((np.random.rand(2) - 1) * 2) * \
                np.asarray([w, h]) * self.t_stdv

            # Rotation
            rot = np.eye(3)
            theta = np.random.vonmises(0.0, self.r_kappa)
            rot[0:2, 0:2] = [[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]]

            # print(rot)

            matrix = matrix @ center @ rot @ uncenter

            # Shear1
            theta1 = np.random.vonmises(0.0, self.sh_kappa)

            shear1 = np.eye(3)
            shear1[0, 1] = theta1

            # print(shear1)

            matrix = matrix @ center @ shear1 @ uncenter

            # Shear2
            theta2 = np.random.vonmises(0.0, self.sh_kappa)

            shear2 = np.eye(3)
            shear2[1, 0] = theta2

            # print(shear2)

            matrix = matrix @ center @ shear2 @ uncenter

            # Scale
            scale = np.eye(3)
            scale[0, 0], scale[1, 1] = np.exp(np.random.rand(2) * self.sc_stdv)

            # print(scale)

            matrix = matrix @ center @ scale @ uncenter

            return AffineTransform(matrix)

        return T.NoOpTransform()


class RandomTranslation(T.Augmentation):
    def __init__(self, prob=0.5, t_stdv=0.02) -> None:
        super().__init__()
        self.prob = prob
        self.t_stdv = t_stdv

    def get_transform(self, image) -> T.Transform:
        if self._rand_range() < self.prob:
            h, w = image.shape[:2]

            matrix = np.eye(3)

            # Translation
            matrix[0:2, 2] = ((np.random.rand(2) - 1) * 2) * \
                np.asarray([w, h]) * self.t_stdv

            # print(matrix)

            return AffineTransform(matrix)

        return T.NoOpTransform()


class RandomRotation(T.Augmentation):
    def __init__(self, prob=0.5, r_kappa=30) -> None:
        super().__init__()
        self.prob = prob
        self.r_kappa = r_kappa

    def get_transform(self, image) -> T.Transform:
        if self._rand_range() < self.prob:
            h, w = image.shape[:2]

            center = np.eye(3)
            center[:2, 2:] = np.asarray([w, h])[:, None] / 2

            # print(center)

            uncenter = np.eye(3)
            uncenter[:2, 2:] = -1 * np.asarray([w, h])[:, None] / 2

            # print(uncenter)

            matrix = np.eye(3)

            # Rotation
            rot = np.eye(3)
            theta = np.random.vonmises(0.0, self.r_kappa)
            rot[0:2, 0:2] = [[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]]

            # print(rot)

            # matrix = uncenter @ rot @ center @ matrix
            matrix = matrix @ center @ rot @ uncenter

            # print(matrix)

            return AffineTransform(matrix)

        return T.NoOpTransform()


class RandomShear(T.Augmentation):
    def __init__(self, prob=0.5, sh_kappa=20) -> None:
        super().__init__()
        self.prob = prob
        self.sh_kappa = sh_kappa

    def get_transform(self, image) -> T.Transform:
        if self._rand_range() < self.prob:
            h, w = image.shape[:2]

            center = np.eye(3)
            center[:2, 2:] = np.asarray([w, h])[:, None] / 2

            uncenter = np.eye(3)
            uncenter[:2, 2:] = -1 * np.asarray([w, h])[:, None] / 2

            matrix = np.eye(3)

            # Shear1
            theta1 = np.random.vonmises(0.0, self.sh_kappa)

            shear1 = np.eye(3)
            shear1[0, 1] = theta1

            # print(shear1)

            matrix = matrix @ center @ shear1 @ uncenter

            # Shear2
            theta2 = np.random.vonmises(0.0, self.sh_kappa)

            shear2 = np.eye(3)
            shear2[1, 0] = theta2

            # print(shear2)

            matrix = matrix @ center @ shear2 @ uncenter

            return AffineTransform(matrix)

        return T.NoOpTransform()


class RandomScale(T.Augmentation):
    def __init__(self, prob=0.5, sc_stdv=0.12) -> None:
        super().__init__()
        self.prob = prob
        self.sc_stdv = sc_stdv

    def get_transform(self, image) -> T.Transform:
        if self._rand_range() < self.prob:
            h, w = image.shape[:2]

            center = np.eye(3)
            center[:2, 2:] = np.asarray([w, h])[:, None] / 2

            uncenter = np.eye(3)
            uncenter[:2, 2:] = -1 * np.asarray([w, h])[:, None] / 2

            matrix = np.eye(3)

            # Scale
            scale = np.eye(3)
            scale[0, 0], scale[1, 1] = np.exp(np.random.rand(2) * self.sc_stdv)

            # print(scale)

            matrix = matrix @ center @ scale @ uncenter

            return AffineTransform(matrix)

        return T.NoOpTransform()


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testing the image augmentation and ")
    parser.add_argument("-i", "--input", help="Input file",
                        required=True, type=str)

    args = parser.parse_args()
    return args


def test(args) -> None:
    import cv2
    from pathlib import Path
    from PIL import Image

    input_path = Path(args.input)

    if not input_path.is_file():
        raise FileNotFoundError(f"Image {input_path} not found")

    print(f"Loading image {input_path}")
    image = cv2.imread(str(input_path))

    resize = T.ResizeShortestEdge((640, 672, 704, 736, 768, 800),
                                  max_size=1333, sample_style="choice", interp=Image.BICUBIC)
    elastic = RandomElastic(prob=1)

    affine = RandomAffine(prob=1)
    translation = RandomTranslation(prob=1)
    rotation = RandomRotation(prob=1)
    shear = RandomShear(prob=1)
    scale = RandomScale(prob=1)

    # augs = T.AugmentationList([resize, elastic, affine])

    # augs = T.AugmentationList([translation, rotation, shear, scale])
    augs = T.AugmentationList([affine])
    # augs = T.AugmentationList([translation])
    # augs = T.AugmentationList([rotation])
    # augs = T.AugmentationList([shear])
    # augs = T.AugmentationList([scale])

    input_augs = T.AugInput(image)

    transforms = augs(input_augs)

    im = Image.fromarray(image)
    im.show("Original")

    im = Image.fromarray(input_augs.image)
    im.show("Transformed")


if __name__ == "__main__":
    args = get_arguments()
    test(args)
