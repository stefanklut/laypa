# Based on: distinctipy https://github.com/alan-turing-institute/distinctipy

import random
from typing import Optional

import numpy as np

# pre-define interesting colours/points at corners, edges, faces and interior of
# r,g,b cube

CORNERS = [
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
]

MID_FACE = [
    (0, 127, 0),
    (0, 0, 127),
    (0, 255, 127),
    (0, 127, 255),
    (0, 127, 127),
    (127, 0, 0),
    (127, 127, 0),
    (127, 255, 0),
    (127, 0, 127),
    (127, 0, 255),
    (127, 255, 127),
    (127, 255, 255),
    (127, 127, 255),
    (255, 127, 0),
    (255, 0, 127),
    (255, 127, 127),
    (255, 255, 127),
    (255, 127, 255),
]

INTERIOR = [
    (127, 127, 127),
    (191, 127, 127),
    (63, 127, 127),
    (127, 191, 127),
    (127, 63, 127),
    (127, 127, 191),
    (127, 127, 63),
]

POINTS_OF_INTEREST = CORNERS + MID_FACE + INTERIOR
_SEED_MAX = int(2**32 - 1)


def _ensure_rng(rng):
    """
    Returns a random.Random state based on the input
    """
    if rng is None:
        rng = random._inst
    elif isinstance(rng, int):
        rng = random.Random(int(rng) % _SEED_MAX)
    elif isinstance(rng, float):
        rng = float(rng)
        # Coerce the float into an integer
        a, b = rng.as_integer_ratio()
        if b == 1:
            seed = a
        else:
            s = max(a.bit_length(), b.bit_length())
            seed = (b << s) | a
        rng = random.Random(seed % _SEED_MAX)
    elif isinstance(rng, random.Random):
        rng = rng
    else:
        raise TypeError(type(rng))
    return rng


def color_distance_squared(color1: tuple[int, int, int], color2: tuple[int, int, int]) -> float:
    """
    Calculate the distance squared between two colors in RGB space, see https://www.compuphase.com/cmetric.htm

    Args:
        color1 (np.ndarray): first color in RGB
        color2 (np.ndarray): second color in RGB

    Returns:
        float: distance between the two colors
    """

    r1, g1, b1 = color1
    r2, g2, b2 = color2

    mean_r = (r1 + r2) / 2
    delta_r = (r1 - r2) ** 2
    delta_g = (g1 - g2) ** 2
    delta_b = (b1 - b2) ** 2

    distance = (2 + mean_r / 256) * delta_r + 4 * delta_g + (2 - (255 - mean_r) / 256) * delta_b

    return distance


def distinct_grayscale(exclude_colors: list[int], n_attempts=1000, rng=None) -> int:
    """
    Find a grayscale color that is distinct from the given colors

    Args:
        exclude_colors (list[int]): list of grayscale colors to exclude
        n_attempts (int, optional): number of attempts to find a distinct color. Defaults to 1000.
        rng (optional): random number generator. Defaults to None.

    Returns:
        int: a grayscale color
    """
    rng = _ensure_rng(rng)
    if not exclude_colors:
        return rng.randint(0, 255)

    max_distance = None
    best_color = None

    # try black and white first
    for color in [0, 255]:
        if color not in exclude_colors:
            distance_to_nearest = min([abs(color - c) for c in exclude_colors])
            if max_distance is None or (distance_to_nearest > max_distance):
                max_distance = distance_to_nearest
                best_color = color

    # try n_attempts randomly generated colors (or all if n_attempts > 256)
    if n_attempts > 256:
        number_generator = range(256)
    else:
        number_generator = rng.sample(range(256), n_attempts)
    for color in number_generator:
        if not exclude_colors:
            return color

        else:
            distance_to_nearest = min([abs(color - c) for c in exclude_colors])

            if (not max_distance) or (distance_to_nearest > max_distance):
                max_distance = distance_to_nearest
                best_color = color

    assert best_color is not None, f"Failed to find a distinct color from {exclude_colors}"

    return best_color


def get_random_color(rng=None) -> tuple[int, int, int]:
    """
    Get a random color in RGB space

    Args:
        rng (optional): random number generator. Defaults to None.

    Returns:
        tuple[int, int, int]: a random RGB color
    """
    rng = _ensure_rng(rng)
    return rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)


def distinct_color(exclude_colors: list[tuple[int, int, int]], n_attempts=1000, rng=None) -> tuple[int, int, int]:
    """
    Find a color that is distinct from the given colors

    Args:
        exclude_colors (list[tuple[int, int, int]]): list of RGB colors to exclude
        n_attempts (int, optional): number of attempts to find a distinct color. Defaults to 1000.
        rng (optional): random number generator. Defaults to None.

    Returns:
        tuple[int, int, int]: a distinct RGB color
    """
    rng = _ensure_rng(rng)

    if not exclude_colors:
        return get_random_color(rng=rng)

    max_distance = None
    best_color = None

    # try pre-defined corners, edges, interior points first
    for color in POINTS_OF_INTEREST:
        if color not in exclude_colors:

            distance_to_nearest = min([color_distance_squared(color, c) for c in exclude_colors])

            if max_distance is None or (distance_to_nearest > max_distance):
                max_distance = distance_to_nearest
                best_color = color

    # try n_attempts randomly generated colors
    for _ in range(n_attempts):
        color = get_random_color(rng=rng)

        if not exclude_colors:
            return color

        else:

            distance_to_nearest = min([color_distance_squared(color, c) for c in exclude_colors])

            if (not max_distance) or (distance_to_nearest > max_distance):
                max_distance = distance_to_nearest
                best_color = color

    assert best_color is not None, f"Failed to find a distinct color from {exclude_colors}"

    return best_color


def n_distinct_colors(
    n_colors: int,
    exclude_colors: Optional[list[tuple[int, int, int]] | list[int]] = None,
    return_excluded=False,
    n_attempts=1000,
    grayscale: bool = False,
    rng=None,
) -> list[tuple[int, int, int]] | list[int]:
    """
    Generate n distinct colors

    Args:
        n_colors (int): number of colors to generate
        exclude_colors (Optional[list[tuple[int, int, int]  |  int]], optional): colors to exclude. Defaults to None.
        return_excluded (bool, optional): flag indicating whether to return the excluded colors. Defaults to False.
        n_attempts (int, optional): number of attempts to generate a distinct color. Defaults to 1000.
        grayscale (bool, optional): flag indicating whether to generate grayscale colors. Defaults to False.
        rng (_type_, optional): random number generator. Defaults to None.

    Returns:
        list[tuple[int, int, int] | int]: list of distinct colors
    """
    rng = _ensure_rng(rng)

    if grayscale:
        assert n_colors <= 256, "Grayscale only supports 256 colors"
        if exclude_colors is None:
            exclude_colors = [0]
        else:
            assert isinstance(exclude_colors, list), "Expected list of grayscale colors"
            assert isinstance(exclude_colors[0], int), "Expected grayscale colors"
        output_colors = exclude_colors.copy()
        for i in range(n_colors):
            output_colors.append(distinct_grayscale(exclude_colors, n_attempts=n_attempts, rng=rng))  # type: ignore
    else:
        assert n_colors <= 256**3, "RGB only supports 256^3 colors"
        if exclude_colors is None:
            exclude_colors = [(0, 0, 0), (255, 255, 255)]
        else:
            assert isinstance(exclude_colors, list), "Expected list of RGB colors"
            assert isinstance(exclude_colors[0], tuple), "Expected RGB colors"
            assert len(exclude_colors[0]) == 3, "Expected RGB colors"
        output_colors = exclude_colors.copy()
        for i in range(n_colors):
            output_colors.append(distinct_color(exclude_colors, n_attempts=n_attempts, rng=rng))  # type: ignore

    if return_excluded:
        return output_colors
    else:
        return output_colors[len(exclude_colors) :]


if __name__ == "__main__":
    colors = n_distinct_colors(10, grayscale=True, rng=0)
    print(colors)
    colors = n_distinct_colors(10, grayscale=True, rng=0)
    print(colors)
    colors = n_distinct_colors(11, grayscale=True, rng=0)
    print(colors)
    colors = n_distinct_colors(14, grayscale=True, rng=0)
    print(colors)
    colors = n_distinct_colors(10, grayscale=True, rng=5)
    print(colors)

    colors = n_distinct_colors(10, grayscale=False, rng=0)
    print(colors)
    colors = n_distinct_colors(10, grayscale=False, rng=0)
    print(colors)
    colors = n_distinct_colors(11, grayscale=False, rng=0)
    print(colors)
    colors = n_distinct_colors(14, grayscale=False, rng=0)
    print(colors)
    colors = n_distinct_colors(10, grayscale=False, rng=5)
    print(colors)
