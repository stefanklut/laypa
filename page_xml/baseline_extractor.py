import itertools
import logging
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from utils.logging_utils import get_logger_name

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xml_regions import XMLRegions

logger = logging.getLogger(get_logger_name())


## Old method
def round_up(array: np.ndarray):
    return np.floor(array + 0.5)


def baseline_converter(
    image: np.ndarray,
    minimum_width: int = 15,
    minimum_height: int = 3,
    step: int = 50,
):
    output = cv2.connectedComponentsWithStats(image, connectivity=8)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    baselines = []

    for i in range(1, num_labels):
        x_offset, y_offset, width, height, area = stats[i]
        sub_image = labels[y_offset : y_offset + height, x_offset : x_offset + width]
        sub_image = (sub_image == i).astype(np.uint8)

        baseline = extract_baseline_v2(sub_image, "test.xml", (x_offset, y_offset), minimum_height, step)
        if len(baseline) < 2:
            continue
        baseline = cv2.approxPolyDP(np.array(baseline, dtype=np.float32), 1, False).reshape(-1, 2)

        if np.max(baseline[:, 0]) - np.min(baseline[:, 0]) < minimum_width:
            continue
        baselines.append(baseline)

    return baselines


def extract_baseline(baseline_mat: np.ndarray, xml_file: str, offset: tuple[int, int], minimum_height: int, step: int = 50):
    baseline = []
    pixel_counter = -1
    merged_line_detected = False
    point = None
    minimum_height = np.minimum(minimum_height, 1)

    for i in range(baseline_mat.shape[1]):
        merged_line_detected_step1 = False
        sum_ = 0
        counter = 0
        for j in range(baseline_mat.shape[0]):
            pixel_value = baseline_mat[j, i]
            if pixel_value:
                sum_ += j
                counter += 1
                if merged_line_detected_step1:
                    merged_line_detected = True
            else:
                if counter > 0:
                    merged_line_detected_step1 = True

        if counter < minimum_height:
            continue

        pixel_counter += 1
        if counter > 1:
            sum_ /= counter

        point = (i + offset[0], sum_ + offset[1])
        if pixel_counter % step == 0:
            baseline.append(point)

    if pixel_counter % step != 0:
        baseline.append(point)

    if merged_line_detected:
        logger.warning(f"mergedLineDetected: {xml_file}")

    return baseline


def extract_baseline_v2(baseline_mat: np.ndarray, xml_file: str, offset: tuple[int, int], minimum_height: int, step: int = 50):
    baseline = []
    pixel_counter = -1
    merged_line_detected = False
    point = None
    minimum_height = np.minimum(minimum_height, 1)

    for i in range(baseline_mat.shape[1]):
        height_points = np.where(baseline_mat[:, i])[0]
        if height_points.size > 1:
            merged_line_detected = merged_line_detected or np.any(height_points[0:-1] + 1 != height_points[1:])

        if height_points.size < minimum_height:
            continue

        pixel_counter += 1

        if pixel_counter % step == 0:
            point = (i + offset[0], np.mean(height_points) + offset[1])
            baseline.append(point)

    if pixel_counter % step != 0:
        if not height_points.size < minimum_height:
            point = (i + offset[0], np.mean(height_points) + offset[1])
        baseline.append(point)

    if merged_line_detected:
        logger.warning(f"mergedLineDetected: {xml_file}")

    return baseline


def test(image: np.ndarray, minimum_height: int = 3, step: int = 50):
    output = cv2.connectedComponentsWithStats(image, connectivity=8)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    baselines = []

    for i in range(1, num_labels):
        x_offset, y_offset, width, height, area = stats[i]
        print(x_offset, y_offset, width, height, area)
        sub_image = labels[y_offset : y_offset + height, x_offset : x_offset + width]
        sub_image = (sub_image == i).astype(np.uint8)

        if area < 2:
            continue

        print(extract_baseline_v3(sub_image, "test.xml", (x_offset, y_offset), minimum_height, step))


def extract_baseline_v3(baseline_mat: np.ndarray, xml_file: str, offset: tuple[int, int], minimum_height: int, step: int = 50):
    baseline = []

    import scipy.ndimage as ndi
    import skimage.graph as graph
    import skimage.morphology as morph

    skeleton_mat = morph.skeletonize(baseline_mat).astype(np.uint8)
    edge = ndi.convolve(skeleton_mat, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode="constant", cval=0)
    edge = np.logical_and(edge == 1, skeleton_mat).astype(np.uint8)

    longest_path = 0
    longest_path_points = None
    for (i1, j1), (i2, j2) in itertools.combinations(np.argwhere(edge), 2):
        route, cost = graph.route_through_array(np.where(skeleton_mat, 1, np.inf), (i1, j1), (i2, j2), fully_connected=True)

        if cost > longest_path:
            longest_path = cost
            longest_path_points = route

    if longest_path_points is None:
        import matplotlib.pyplot as plt

        plt.imshow(baseline_mat, cmap="gray")
        plt.show()

        plt.imshow(skeleton_mat, cmap="gray")
        plt.show()

        plt.imshow(edge, cmap="gray")
        plt.show()

    return longest_path_points


def top_bottom_overlap(image: np.ndarray, top: int, bottom: int):
    top_image = (image == top).astype(np.uint8)
    bottom_image = (image == bottom).astype(np.uint8)
    dilated_top = cv2.dilate(top_image, np.ones((3, 3), np.uint8))
    dilated_bottom = cv2.dilate(bottom_image, np.ones((3, 3), np.uint8))
    overlap = np.logical_and(dilated_top, dilated_bottom)
    overlap = np.logical_and(overlap, image != 0)
    return overlap.astype(np.uint8)


def overlap_pixels(image1: np.ndarray, image2: np.ndarray):
    return np.any(np.logical_and(image1, image2))


def overlap_rectangles(rectangle1: tuple[int, int, int, int], rectangle2: tuple[int, int, int, int]):
    x1, y1, w1, h1 = rectangle1
    x2, y2, w2, h2 = rectangle2

    # If one rectangle is empty, there is no overlap
    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        return False
    # If one rectangle is to the left or right of the other, there is no overlap
    if x1 + w1 < x2 or x2 + w2 < x1:
        return False
    # If one rectangle is above or below the other, there is no overlap
    if y1 + h1 < y2 or y2 + h2 < y1:
        return False
    return True


def combine_rectangles(rectangles: Iterable[tuple[int, int, int, int]]):
    x = min(rectangle[0] for rectangle in rectangles)
    y = min(rectangle[1] for rectangle in rectangles)
    w = max(rectangle[0] + rectangle[2] for rectangle in rectangles) - x
    h = max(rectangle[1] + rectangle[3] for rectangle in rectangles) - y
    return x, y, w, h


def get_sub_image(image: np.ndarray, rectangle: tuple[int, int, int, int]):
    x, y, w, h = rectangle
    return image[y : y + h, x : x + w]


def most_likely_orientation(overlap_image: np.ndarray, top_image: np.ndarray, bottom_image: np.ndarray, step: int = 1):
    assert step > 0, "Step must be greater than 0"
    assert 360 % step == 0, "Step must be a divisor of 360"
    assert overlap_image.shape == top_image.shape == bottom_image.shape, "All images must have the same shape"

    (h, w) = overlap_image.shape[:2]

    max_dimension = max(h, w)

    translated_overlap_image = np.zeros((max_dimension * 2, max_dimension * 2), dtype=np.uint8)
    translated_overlap_image[
        max_dimension - h // 2 : max_dimension - h // 2 + h,
        max_dimension - w // 2 : max_dimension - w // 2 + w,
    ] = overlap_image

    translated_top_image = np.zeros((max_dimension * 2, max_dimension * 2), dtype=np.uint8)
    translated_top_image[
        max_dimension - h // 2 : max_dimension - h // 2 + h,
        max_dimension - w // 2 : max_dimension - w // 2 + w,
    ] = top_image

    translated_bottom_image = np.zeros((max_dimension * 2, max_dimension * 2), dtype=np.uint8)
    translated_bottom_image[
        max_dimension - h // 2 : max_dimension - h // 2 + h,
        max_dimension - w // 2 : max_dimension - w // 2 + w,
    ] = bottom_image

    best_orientation_score = None
    output = None

    for orientation in range(0, 360, step):
        rotation_matrix = cv2.getRotationMatrix2D((max_dimension, max_dimension), orientation, 1)
        rotated_overlap_image = cv2.warpAffine(
            translated_overlap_image, rotation_matrix, translated_overlap_image.shape, flags=cv2.INTER_NEAREST
        )

        rotated_top_image = cv2.warpAffine(
            translated_top_image, rotation_matrix, translated_top_image.shape, flags=cv2.INTER_NEAREST
        )

        rotated_bottom_image = cv2.warpAffine(
            translated_bottom_image, rotation_matrix, translated_bottom_image.shape, flags=cv2.INTER_NEAREST
        )

        thickness = 5
        top_above_overlap = 0
        bottom_below_overlap = 0
        for i in range(rotated_overlap_image.shape[1]):
            height_points = np.where(rotated_overlap_image[:, i])[0]
            if height_points.size == 0:
                continue
            middle_overlap = np.round(np.mean(height_points)).astype(int)
            top_above_overlap += np.sum(rotated_top_image[middle_overlap - thickness : middle_overlap, i])
            bottom_below_overlap += np.sum(rotated_bottom_image[middle_overlap : middle_overlap + thickness, i])

        score = top_above_overlap + bottom_below_overlap
        if best_orientation_score is None or score > best_orientation_score:
            best_orientation_score = score
            output = (
                rotated_overlap_image,
                rotated_top_image,
                rotated_bottom_image,
                rotation_matrix,
                (max_dimension - w // 2, max_dimension - h // 2),
            )

    return output


def top_bottom_converter(
    image: np.ndarray,
    minimum_width: int = 15,
    minimum_height: int = 3,
    step: int = 50,
):
    top_value = 1
    bottom_value = 2

    top_bottom_overlap_image = top_bottom_overlap(image, top_value, bottom_value)
    output_overlap = cv2.connectedComponentsWithStats(top_bottom_overlap_image, connectivity=8)
    num_labels_overlap = output_overlap[0]
    labels_overlap = output_overlap[1]
    stats_overlap = output_overlap[2]
    centroids_overlap = output_overlap[3]

    output_top = cv2.connectedComponentsWithStats((image == top_value).astype(np.uint8), connectivity=8)
    num_labels_top = output_top[0]
    labels_top = output_top[1]
    stats_top = output_top[2]
    centroids_top = output_top[3]
    output_bottom = cv2.connectedComponentsWithStats((image == bottom_value).astype(np.uint8), connectivity=8)
    num_labels_bottom = output_bottom[0]
    labels_bottom = output_bottom[1]
    stats_bottom = output_bottom[2]
    centroids_bottom = output_bottom[3]

    baselines = []

    for i in range(1, num_labels_overlap):
        x_offset_overlap, y_offset_overlap, width_overlap, height_overlap, area_overlap = stats_overlap[i]
        rectangle_overlap = (x_offset_overlap, y_offset_overlap, width_overlap, height_overlap)

        assigned_top_labels = []
        assigned_bottom_labels = []
        for j in range(1, num_labels_top):
            x_offset_top, y_offset_top, width_top, height_top, area_top = stats_top[j]

            rectangle_top = (x_offset_top, y_offset_top, width_top, height_top)
            if not overlap_rectangles(rectangle_overlap, rectangle_top):
                continue

            rectangle_combined = combine_rectangles((rectangle_overlap, rectangle_top))

            sub_labels_overlap = get_sub_image(labels_overlap, rectangle_combined)
            sub_labels_overlap = sub_labels_overlap == i

            sub_labels_top = get_sub_image(labels_top, rectangle_combined)
            sub_labels_top = sub_labels_top == j

            if overlap_pixels(sub_labels_overlap, sub_labels_top):
                assigned_top_labels.append(j)

        for j in range(1, num_labels_bottom):
            x_offset_bottom, y_offset_bottom, width_bottom, height_bottom, area_bottom = stats_bottom[j]
            rectangle_bottom = (x_offset_bottom, y_offset_bottom, width_bottom, height_bottom)
            if not overlap_rectangles(rectangle_overlap, rectangle_bottom):
                continue

            rectangle_combined = combine_rectangles((rectangle_overlap, rectangle_bottom))

            sub_labels_overlap = get_sub_image(labels_overlap, rectangle_combined)
            sub_labels_overlap = sub_labels_overlap == i

            sub_labels_bottom = get_sub_image(labels_bottom, rectangle_combined)
            sub_labels_bottom = sub_labels_bottom == j

            if overlap_pixels(sub_labels_overlap, sub_labels_bottom):
                assigned_bottom_labels.append(j)

        assert assigned_top_labels and assigned_bottom_labels, "No top or bottom label assigned"

        rectangles = []
        for top_label in assigned_top_labels:
            x_offset_top, y_offset_top, width_top, height_top, area_top = stats_top[top_label]
            rectangle_top = (x_offset_top, y_offset_top, width_top, height_top)
            rectangles.append(rectangle_top)

        for bottom_label in assigned_bottom_labels:
            x_offset_bottom, y_offset_bottom, width_bottom, height_bottom, area_bottom = stats_bottom[bottom_label]
            rectangle_bottom = (x_offset_bottom, y_offset_bottom, width_bottom, height_bottom)
            rectangles.append(rectangle_bottom)

        rectangle_combined = combine_rectangles(rectangles)
        top_bottom_overlap_sub_image = get_sub_image(labels_overlap, rectangle_combined)
        top_bottom_overlap_sub_image = top_bottom_overlap_sub_image == i
        top_sub_image = get_sub_image(labels_top, rectangle_combined)
        top_sub_image = np.isin(top_sub_image, assigned_top_labels)
        bottom_sub_image = get_sub_image(labels_bottom, rectangle_combined)
        bottom_sub_image = np.isin(bottom_sub_image, assigned_bottom_labels)

        best_orientation = most_likely_orientation(
            top_bottom_overlap_sub_image,
            top_sub_image,
            bottom_sub_image,
            45,
        )

        if best_orientation is None:
            raise ValueError("No best orientation found")

        rotated_overlap_image, rotated_top_image, rotated_bottom_image, rotation_matrix, offset = best_orientation

        baseline = extract_baseline_v2(rotated_overlap_image, "test.xml", (0, 0), 1, step)
        if len(baseline) < 2:
            continue
        baseline = cv2.approxPolyDP(np.array(baseline, dtype=np.float32), minimum_height, False).reshape(-1, 2)

        if np.max(baseline[:, 0]) - np.min(baseline[:, 0]) < minimum_width:
            continue

        inv_rotation_matrix = np.eye(3)
        inv_rotation_matrix[:2] = rotation_matrix
        inv_rotation_matrix = np.linalg.inv(inv_rotation_matrix)

        baseline = cv2.transform(baseline[:, None, :], inv_rotation_matrix)[:, 0, :2]
        baseline += np.asarray([rectangle_combined[0], rectangle_combined[1]]) - offset

        baselines.append(baseline)

    return baselines


def test2(image: np.ndarray):
    return top_bottom_converter(image)


def image_to_baselines(
    image: np.ndarray,
    xml_regions: XMLRegions,
    minimum_width: int = 15,
    minimum_height: int = 3,
    step: int = 50,
):
    if xml_regions.mode == "baseline":
        return baseline_converter(image, minimum_width, minimum_height, step)
    if xml_regions.mode == "top_bottom":
        return top_bottom_converter(image, minimum_width, minimum_height, step)
    else:
        raise ValueError(f"Unsupported mode: {xml_regions.mode}")


if __name__ == "__main__":
    image = cv2.imread("/home/stefan/Downloads/276/page/NL-HaNA_2.01.01.01_276_0011.png", cv2.IMREAD_GRAYSCALE)

    print(test(image))
