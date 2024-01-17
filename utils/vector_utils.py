import cv2
import numpy as np


def point_line_segment_distance(line_segments: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Calculate the distance between points and line segments.

    Args:
        line_segments (np.ndarray): Array of line segment coordinates.
        points (np.ndarray): Array of point coordinates.

    Returns:
        np.ndarray: Array of distances between points and line segments.
    """
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    line_norm = np.linalg.norm(line_vector, axis=1)
    start_to_points_vector = points[:, np.newaxis, :] - line_starts
    start_to_points_norm = np.linalg.norm(start_to_points_vector, axis=2)
    end_to_points_vector = points[:, np.newaxis, :] - line_ends
    end_to_points_norm = np.linalg.norm(end_to_points_vector, axis=2)

    cross_product = start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]
    line_to_points_norm = np.abs(cross_product) / line_norm

    dot_product = start_to_points_vector[:, :, 0] * line_vector[:, 0] + start_to_points_vector[:, :, 1] * line_vector[:, 1]
    dot_product_projection = dot_product / (line_norm * line_norm)

    mask_less_0 = dot_product_projection <= 0
    mask_more_1 = dot_product_projection >= 1
    mask_between_0_1 = np.logical_not(np.logical_or(mask_less_0, mask_more_1))

    distances = np.zeros((len(points), len(line_segments) - 1))

    distances[mask_less_0] = start_to_points_norm[mask_less_0]
    distances[mask_more_1] = end_to_points_norm[mask_more_1]
    distances[mask_between_0_1] = line_to_points_norm[mask_between_0_1]

    return distances


def point_at_start_or_end_assignment(line_segments: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Find and remove points that are at the start or end of a line segment.

    Args:
        line_segments (np.ndarray): Array of line segment coordinates.
        points (np.ndarray): Array of point coordinates.

    Returns:
        np.ndarray: Array of labels indicating if points are at the start or end of a line segment.
    """

    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    line_norm = np.linalg.norm(line_vector, axis=1)

    # Remove zero length lines
    non_zero = line_norm != 0

    # HACK If baseline has lenght 0, Do not remove any points
    if not np.any(non_zero):
        return np.zeros(len(points))
    else:
        line_vector = line_vector[non_zero]
        line_ends = line_ends[non_zero]
        line_starts = line_starts[non_zero]
        line_norm = line_norm[non_zero]

    start_to_points_vector = points[:, np.newaxis, :] - line_starts
    start_to_points_norm = np.linalg.norm(start_to_points_vector, axis=2)
    end_to_points_vector = points[:, np.newaxis, :] - line_ends
    end_to_points_norm = np.linalg.norm(end_to_points_vector, axis=2)

    cross_product = start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]
    line_to_points_norm = np.abs(cross_product) / line_norm

    dot_product = start_to_points_vector[:, :, 0] * line_vector[:, 0] + start_to_points_vector[:, :, 1] * line_vector[:, 1]
    dot_product_projection = dot_product / (line_norm * line_norm)

    mask_less_0 = dot_product_projection <= 0
    mask_more_1 = dot_product_projection >= 1
    mask_between_0_1 = np.logical_not(np.logical_or(mask_less_0, mask_more_1))

    distances = np.zeros((len(points), len(line_vector)))

    distances[mask_less_0] = start_to_points_norm[mask_less_0]
    distances[mask_more_1] = end_to_points_norm[mask_more_1]
    distances[mask_between_0_1] = line_to_points_norm[mask_between_0_1]

    min_values = np.min(distances, axis=1)
    is_min_value = distances == min_values[:, np.newaxis]

    label = np.zeros(len(points))

    closest_to_start = np.any(is_min_value[:, [0]], axis=1)
    closest_to_end = np.any(is_min_value[:, [-1]], axis=1)

    for i, check in enumerate(closest_to_start):
        if check:
            if mask_less_0[i, 0]:
                label[i] = 1
    for i, check in enumerate(closest_to_end):
        if check:
            if mask_more_1[i, -1]:
                label[i] = 1

    return label


def consecutive_booleans(array: np.ndarray) -> np.ndarray:
    """
    Find consecutive True values in a boolean array.

    Args:
        array (np.ndarray): Boolean array.

    Returns:
        np.ndarray: Boolean array indicating consecutive True values.
    """
    overlap = np.logical_and(array[1:], array[:-1])
    return overlap


def point_line_segment_assignment(line_segments: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Assign points to line segments based on minimum distance.

    Args:
        line_segments (np.ndarray): Array of line segment coordinates.
        points (np.ndarray): Array of point coordinates.

    Returns:
        np.ndarray: Array of labels indicating the assigned line segment for each point.
    """
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    line_norm = np.linalg.norm(line_vector, axis=1)

    # Find closest line segment
    start_to_points_vector = points[:, np.newaxis, :] - line_starts
    start_to_points_norm = np.linalg.norm(start_to_points_vector, axis=2)
    end_to_points_vector = points[:, np.newaxis, :] - line_ends
    end_to_points_norm = np.linalg.norm(end_to_points_vector, axis=2)

    cross_product = start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]
    line_to_points_norm = np.abs(cross_product) / line_norm

    dot_product = start_to_points_vector[:, :, 0] * line_vector[:, 0] + start_to_points_vector[:, :, 1] * line_vector[:, 1]
    dot_product_projection = dot_product / (line_norm * line_norm)

    mask_less_0 = dot_product_projection <= 0
    mask_more_1 = dot_product_projection >= 1
    mask_between_0_1 = np.logical_not(np.logical_or(mask_less_0, mask_more_1))

    distances = np.zeros((len(points), len(line_segments) - 1))

    distances[mask_less_0] = start_to_points_norm[mask_less_0]
    distances[mask_more_1] = end_to_points_norm[mask_more_1]
    distances[mask_between_0_1] = line_to_points_norm[mask_between_0_1]

    min_values = np.min(distances, axis=1)

    # Find points that are have the same minimum distance to multiple lines
    is_min_value = distances == min_values[:, np.newaxis]
    min_count = np.sum(is_min_value, axis=1)
    min_occurs_more_than_once = min_count > 1

    label = np.zeros(min_occurs_more_than_once.shape)

    for i, check in enumerate(min_occurs_more_than_once):
        if check:
            # If the overlap is consecutive, assign the based on a middle vector that splits the overlap
            overlap_bool = consecutive_booleans(is_min_value[i])
            overlap_location = np.argwhere(overlap_bool)
            for j in overlap_location:
                middle_vector = line_vector[j] / line_norm[j] - line_vector[j + 1] / line_norm[j + 1]

                cross_product_1 = (
                    end_to_points_vector[i, j, 0] * middle_vector[:, 1] - end_to_points_vector[i, j, 1] * middle_vector[:, 0]
                )
                cross_product_2 = -line_vector[j, 0] * middle_vector[:, 1] + line_vector[j, 1] * middle_vector[:, 0]

                if cross_product_1 == 0 or cross_product_2 == 0:
                    label[i] = np.random.choice([j, j + 1])
                elif np.sign(cross_product_1) == np.sign(cross_product_2):
                    label[i] = j
                else:
                    label[i] = j + 1

            # label[i] = 5
        else:
            # If there is a single minimum distance, assign the point to the line
            label[i] = np.argwhere(is_min_value[i])

    return label


def point_line_segment_side(line_segments: np.ndarray, points: np.ndarray) -> tuple:
    """
    Determine which side of the line segments points are located.

    Args:
        line_segments (np.ndarray): Array of line segment coordinates.
        points (np.ndarray): Array of point coordinates.

    Returns:
        tuple: Tuple of boolean arrays indicating if points are left, right, or on the line segments.
    """
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    start_to_points_vector = points[:, np.newaxis, :] - line_starts

    cross_product = start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]

    # Get the sign of the cross product which indicates if the point is left or right of the line
    left_of_line_vector = cross_product > 0
    right_of_line_vector = cross_product < 0
    on_line_vector = cross_product == 0
    return (left_of_line_vector, right_of_line_vector, on_line_vector)


def point_top_bottom_assignment(line_segments: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Assign points to top or bottom sides of line segments based on minimum distance.

    Args:
        line_segments (np.ndarray): Array of line segment coordinates.
        points (np.ndarray): Array of point coordinates.

    Returns:
        np.ndarray: Array of labels indicating the assigned side (top or bottom) for each point.
    """
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    line_norm = np.linalg.norm(line_vector, axis=1)

    # Remove zero length lines
    non_zero = line_norm != 0

    # HACK If baseline has lenght 0, assume top bottom from the pages perspective
    if not np.any(non_zero):
        line_starts = line_segments[0:1]
        line_ends = line_starts[0:1] + np.asarray([1, 0])
        line_vector = line_ends - line_starts
        line_norm = np.linalg.norm(line_vector, axis=1)
    else:
        line_vector = line_vector[non_zero]
        line_ends = line_ends[non_zero]
        line_starts = line_starts[non_zero]
        line_norm = line_norm[non_zero]

    # Find closest line segment
    start_to_points_vector = points[:, np.newaxis, :] - line_starts
    start_to_points_norm = np.linalg.norm(start_to_points_vector, axis=2)
    end_to_points_vector = points[:, np.newaxis, :] - line_ends
    end_to_points_norm = np.linalg.norm(end_to_points_vector, axis=2)

    cross_product = start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]
    line_to_points_norm = np.abs(cross_product) / line_norm

    dot_product = start_to_points_vector[:, :, 0] * line_vector[:, 0] + start_to_points_vector[:, :, 1] * line_vector[:, 1]
    dot_product_projection = dot_product / (line_norm * line_norm)

    mask_less_0 = dot_product_projection <= 0
    mask_more_1 = dot_product_projection >= 1
    mask_between_0_1 = np.logical_not(np.logical_or(mask_less_0, mask_more_1))

    distances = np.zeros((len(points), len(line_vector)))

    distances[mask_less_0] = start_to_points_norm[mask_less_0]
    distances[mask_more_1] = end_to_points_norm[mask_more_1]
    distances[mask_between_0_1] = line_to_points_norm[mask_between_0_1]

    min_values = np.min(distances, axis=1)

    # Find points that are have the same minimum distance to multiple lines
    is_min_value = distances == min_values[:, np.newaxis]
    min_count = np.sum(is_min_value, axis=1)
    min_occurs_more_than_once = min_count > 1

    label = np.zeros(len(points))

    for i, check in enumerate(min_occurs_more_than_once):
        if check:
            # If there are multiple minimum distances, assign the point to a line segment
            overlap_bool = consecutive_booleans(is_min_value[i])
            if not np.any(overlap_bool):
                # If the overlap is not consecutive, assign a random line
                min_lines = np.nonzero(is_min_value[i])[0]

                # import matplotlib.pyplot as plt
                # print(is_min_value[i], distances[i])
                # line = np.vstack((line_starts, line_ends[-1]))
                # print(line)
                # plt.cla()
                # plt.plot(line[:, 0], line[:, 1], color='b', alpha=0.5)
                # for k in min_lines:
                #     print(line[k:k+2])
                #     plt.plot(line[k:k+2, 0], line[k:k+2, 1], color='g', linewidth=3)
                # plt.scatter(points[i, 0], points[i, 1], color='r')
                # plt.show()
                cross_product_i = cross_product[i, np.random.choice(min_lines)]
            else:
                # If the overlap is consecutive, assign the based on a middle vector that splits the overlap
                cross_product_i = 0
                overlap_location = np.argwhere(overlap_bool)[0]

                for j in overlap_location:
                    # Find what side of the middle vector the points and first line segment are on
                    middle_vector = line_vector[j] / line_norm[j] - line_vector[j + 1] / line_norm[j + 1]

                    cross_product_1 = (
                        end_to_points_vector[i, j, 0] * middle_vector[1] - end_to_points_vector[i, j, 1] * middle_vector[0]
                    )
                    cross_product_2 = -line_vector[j, 0] * middle_vector[1] + line_vector[j, 1] * middle_vector[0]

                    # Select the cross product of the assigned line segment
                    if cross_product_1 == 0 or cross_product_2 == 0:
                        # If the cross product is zero, assign a random line
                        cross_product_i = cross_product[i, np.random.choice([j, j + 1])]
                    elif np.sign(cross_product_1) == np.sign(cross_product_2):
                        # If the point and vector are on the same side, assign the first line
                        cross_product_i = cross_product[i, j]
                    else:
                        # If the point and vector are on different sides, assign the second line
                        cross_product_i = cross_product[i, j + 1]

            # Based on the selected line, assign the point to the top or bottom side
            if cross_product_i == 0:
                label[i] = np.random.choice([0, 1])
            elif cross_product_i > 0:
                label[i] = 1
            else:
                label[i] = 0
        else:
            # If there is a single minimum distance, assign the point to the line and the top or bottom side
            cross_product_i = cross_product[i, np.argwhere(is_min_value[i])]
            if cross_product_i == 0:
                label[i] = np.random.choice([0, 1])
            elif cross_product_i > 0:
                label[i] = 1
            else:
                label[i] = 0

    return label


def draw_lines(lines: np.ndarray, thickness: int) -> np.ndarray:
    """
    Draw lines with specified thickness and assign point labels.

    Args:
        lines (np.ndarray): Array of line segment coordinates.
        thickness (int): Thickness of lines.

    Returns:
        np.ndarray: Image with line segments and assigned point labels.
    """
    height, width = 1000, 1000
    empty_mask = np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        rounded_coords = np.round(line).astype(np.int32)
        cv2.polylines(empty_mask, [rounded_coords.reshape(-1, 1, 2)], False, 1, thickness)
        line_pixel_coords = np.column_stack(np.where(empty_mask == 1))[:, ::-1]
        print(line_pixel_coords.shape)
        empty_mask.fill(0)
        mask[line_pixel_coords[:, 1], line_pixel_coords[:, 0]] = (
            point_at_start_or_end_assignment(line, line_pixel_coords) + 1
        ) * 100

    return mask


if __name__ == "__main__":
    # coordinates = np.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
    # coordinates = np.array([(10, 10)])
    # coordinates = np.random.rand(100,2) * 10
    # X, Y = np.mgrid[0:10:100j, 0:10:100j]
    # coordinates = np.vstack([X.ravel(), Y.ravel()]).T
    # line_segments = np.array([(0, 0), (1, 1), (2,3), (4, 2), (7, 8), (10,10), (5,8)])
    # labels = point_top_bottom_assignment(line_segments, coordinates)

    # import sys
    # sys.exit()
    # result = point_line_segment_distance(line_segments, coordinates)

    # labels = np.argmin(result, axis=1)
    # print(labels.shape)
    # result_ = point_line_segment_side(line_segments, coordinates)
    # labels = np.take_along_axis(result_[0], labels[:,None], axis=1).astype(int)
    # print(labels.shape)

    # import matplotlib.colors
    # import matplotlib.pyplot as plt
    # colors = ['red','green','blue', 'purple', 'black', 'grey', 'yellow']

    # fig = plt.figure(figsize=(8,8))
    # plt.plot(line_segments[..., 0], line_segments[..., 1])
    # plt.scatter(coordinates[..., 0], coordinates[..., 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    # for i, txt in enumerate(np.min(result,axis=1)):
    #     label = labels[i]
    #     plt.annotate(f"{coordinates[i]} {label} {np.round(result[i], 3)}", (coordinates[i, 0], coordinates[i, 1]))
    # plt.show()

    import matplotlib.pyplot as plt

    im = draw_lines(np.array([[(10, 10), (10, 10), (100, 100), (200, 105)]]), 10)
    plt.imshow(im, cmap="gray")
    plt.show()
