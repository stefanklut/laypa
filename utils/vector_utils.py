import numpy as np
import cv2

def point_line_segment_distance(line_segments, points):
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    line_norm = np.linalg.norm(line_vector, axis=1)
    start_to_points_vector = points[:, np.newaxis, :] - line_starts
    start_to_points_norm = np.linalg.norm(start_to_points_vector, axis=2)
    end_to_points_vector = points[:, np.newaxis, :] - line_ends
    end_to_points_norm = np.linalg.norm(end_to_points_vector, axis=2)

    cross_product =  start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]
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

def consecutive_booleans(array):
    overlap = np.logical_and(array[1:], array[:-1])
    return overlap

def point_line_segment_assignment(line_segments, points):
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    line_norm = np.linalg.norm(line_vector, axis=1)
    start_to_points_vector = points[:, np.newaxis, :] - line_starts
    start_to_points_norm = np.linalg.norm(start_to_points_vector, axis=2)
    end_to_points_vector = points[:, np.newaxis, :] - line_ends
    end_to_points_norm = np.linalg.norm(end_to_points_vector, axis=2)

    cross_product =  start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]
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
    is_min_value = distances == min_values[:, np.newaxis]
    
    min_count = np.sum(is_min_value, axis=1)
    min_occurs_more_than_once = min_count > 1
    
    label = np.zeros(min_occurs_more_than_once.shape)
    
    for i, check in enumerate(min_occurs_more_than_once):
        if check:
            overlap_bool = consecutive_booleans(is_min_value[i])
            overlap_location = np.argwhere(overlap_bool)
            for j in overlap_location:
                middle_vector = line_vector[j] / line_norm[j] - line_vector[j+1] / line_norm[j+1]
                
                cross_product_1 =  end_to_points_vector[i, j, 0] * middle_vector[:, 1] - end_to_points_vector[i, j, 1] * middle_vector[:, 0]
                cross_product_2 = -line_vector[j, 0] * middle_vector[:, 1] + line_vector[j, 1] * middle_vector[:, 0]
                
                if cross_product_1 == 0 or cross_product_2 == 0:
                    label[i] = np.random.choice([j, j+1])
                elif np.sign(cross_product_1) == np.sign(cross_product_2):
                    label[i] = j
                else:
                    label[i] = j+1

            # label[i] = 5
        else:
            label[i] = np.argwhere(is_min_value[i])
            
    return label

def point_line_segment_side(line_segments, points):
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    start_to_points_vector = points[:, np.newaxis, :] - line_starts


    cross_product =  start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]
    
    left_of_line_vector = cross_product > 0
    right_of_line_vector = cross_product < 0
    on_line_vector = cross_product == 0
    return (left_of_line_vector, right_of_line_vector, on_line_vector)

def point_top_bottom_assignment(line_segments, points):
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    line_norm = np.linalg.norm(line_vector, axis=1)
    start_to_points_vector = points[:, np.newaxis, :] - line_starts
    start_to_points_norm = np.linalg.norm(start_to_points_vector, axis=2)
    end_to_points_vector = points[:, np.newaxis, :] - line_ends
    end_to_points_norm = np.linalg.norm(end_to_points_vector, axis=2)

    cross_product =  start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]
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
    is_min_value = distances == min_values[:, np.newaxis]
    
    min_count = np.sum(is_min_value, axis=1)
    min_occurs_more_than_once = min_count > 1
    
    label = np.zeros(min_occurs_more_than_once.shape)
    
    for i, check in enumerate(min_occurs_more_than_once):
        if check:
            overlap_bool = consecutive_booleans(is_min_value[i])
            if not np.any(overlap_bool):
                min_lines = np.argwhere(is_min_value[i])
                cross_product_i = cross_product[i, np.random.choice(min_lines)]
            else:
                cross_product_i = 0
                overlap_location = np.argwhere(overlap_bool)[0]
                
                for j in overlap_location:
                    middle_vector = line_vector[j] / line_norm[j] - line_vector[j+1] / line_norm[j+1]
                    
                    cross_product_1 =  end_to_points_vector[i, j, 0] * middle_vector[1] - end_to_points_vector[i, j, 1] * middle_vector[0]
                    cross_product_2 = -line_vector[j, 0] * middle_vector[1] + line_vector[j, 1] * middle_vector[0]
                    
                    if cross_product_1 == 0 or cross_product_2 == 0:
                        cross_product_i = cross_product[i, np.random.choice([j, j+1])]
                    elif np.sign(cross_product_1) == np.sign(cross_product_2):
                        cross_product_i = cross_product[i, j]
                    else:
                        cross_product_i = cross_product[i, j+1]
                        
            if cross_product_i == 0:
                label[i] = np.random.choice([0, 1])
            elif cross_product_i > 0:
                label[i] = 0
            else:
                label[i] = 1
        else:
            cross_product_i = cross_product[i, np.argwhere(is_min_value[i])]
            if cross_product_i == 0:
                label[i] = np.random.choice([0, 1])
            elif cross_product_i > 0:
                label[i] = 0
            else:
                label[i] = 1
            
    return label

def draw_lines(lines, thickness):
    height, width = 1000, 1000
    empty_mask = np.zeros((height, width), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in lines:
        rounded_coords = np.round(line).astype(np.int32)
        cv2.polylines(empty_mask, [rounded_coords.reshape(-1, 1, 2)], False, 1, thickness)
        line_pixel_coords = np.column_stack(np.where(empty_mask == 1))
        print(line_pixel_coords.shape)
        empty_mask.fill(0)
        mask[line_pixel_coords[:, 0], line_pixel_coords[:, 1]] = (point_top_bottom_assignment(line, line_pixel_coords[:, ::-1])+1) * 100

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
    im = draw_lines(np.array([[(10,10), (210,10), (400,60)]]), 10)
    plt.imshow(im)
    plt.show()