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

        baseline = extract_baseline(sub_image, (x_offset, y_offset), minimum_height, "test.xml")
        if len(baseline) < 2:
            continue
        baseline = cv2.approxPolyDP(np.array(baseline, dtype=np.float32), 1, False).reshape(-1, 2)

        if np.max(baseline[:, 0]) - np.min(baseline[:, 0]) < minimum_width:
            continue
        baselines.append(baseline)

    return baselines


def extract_baseline(baseline_mat: np.ndarray, offset: tuple[int, int], minimum_height: int, xml_file: str):
    baseline = []
    pixel_counter = -1
    merged_line_detected = False
    point = None

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
        if pixel_counter % 50 == 0:
            baseline.append(point)

    if pixel_counter % 50 != 0:
        baseline.append(point)

    if merged_line_detected:
        print(f"mergedLineDetected: {xml_file}")

    return baseline


def image_to_baselines(image: np.ndarray, xml_regions: XMLRegions):
    if xml_regions.mode == "baseline":
        return baseline_converter(image)


if __name__ == "__main__":
    image = cv2.imread("/tmp/results/page/NL-HaNA_2.09.09_5_0023.png", cv2.IMREAD_GRAYSCALE)

    baselines = image_to_baselines(image, XMLRegions("baseline", 10))

    for baseline in baselines:
        print(baseline)
