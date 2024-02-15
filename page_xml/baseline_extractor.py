import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xml_regions import XMLRegions


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
        print(f"mergedLineDetected: {xml_file}")

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
        print(f"mergedLineDetected: {xml_file}")

    return baseline


def test(image: np.ndarray, minimum_height: int, step: int = 50):
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

        extract_baseline_v3(sub_image, "test.xml", (x_offset, y_offset), minimum_height, step)


def extract_baseline_v3(baseline_mat: np.ndarray, xml_file: str, offset: tuple[int, int], minimum_height: int, step: int = 50):
    baseline = []

    import scipy.ndimage as ndi
    import skimage.morphology as morph

    skeleton_mat = morph.skeletonize(baseline_mat).astype(np.uint8)
    edge = ndi.convolve(skeleton_mat, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode="constant", cval=0)
    edge = np.logical_and(edge == 1, skeleton_mat).astype(np.uint8)

    # skeleton_mat = cv2.ximgproc.thinning(baseline_mat * 255, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    if np.sum(edge) != 2:

        import matplotlib.pyplot as plt

        plt.imshow(baseline_mat * 255, cmap="gray")
        plt.show()
        plt.imshow(skeleton_mat * 255, cmap="gray")
        plt.show()
        plt.imshow(edge * 255, cmap="gray")
        plt.show()


def image_to_baselines(image: np.ndarray, xml_regions: XMLRegions):
    if xml_regions.mode == "baseline":
        return baseline_converter(image)


if __name__ == "__main__":
    image = cv2.imread("/tmp/results/page/NL-HaNA_2.09.09_5_0023.png", cv2.IMREAD_GRAYSCALE)

    test(image, 3)
