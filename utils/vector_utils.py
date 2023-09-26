import numpy as np

def point_line_segment_distance(line_segments, points):
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    start_to_points_vector = points[:, np.newaxis, :] - line_starts
    start_to_points_norm = np.linalg.norm(start_to_points_vector, axis=2)
    end_to_points_vector = points[:, np.newaxis, :] - line_ends
    end_to_points_norm = np.linalg.norm(end_to_points_vector, axis=2)

    cross_product =  start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]
    line_to_points_norm = np.abs(cross_product) / np.linalg.norm(line_vector, axis=1)
    
    dot_product = start_to_points_vector[:, :, 0] * line_vector[:, 0] + start_to_points_vector[:, :, 1] * line_vector[:, 1]
    dot_product_projection = dot_product / np.linalg.norm(line_vector, axis=1)**2
    
    mask_less_0 = dot_product_projection <= 0
    mask_more_1 = dot_product_projection >= 1
    mask_between_0_1 = np.logical_not(np.logical_or(mask_less_0, mask_more_1))
    
    distances = np.zeros((len(points), len(line_segments) - 1))
    
    distances[mask_less_0] = start_to_points_norm[mask_less_0]
    distances[mask_more_1] = end_to_points_norm[mask_more_1]
    distances[mask_between_0_1] = line_to_points_norm[mask_between_0_1]
    
    return distances

def point_line_segment_assignment(line_segments, points):
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_vector = line_ends - line_starts
    start_to_points_vector = points[:, np.newaxis, :] - line_starts
    start_to_points_norm = np.linalg.norm(start_to_points_vector, axis=2)
    end_to_points_vector = points[:, np.newaxis, :] - line_ends
    end_to_points_norm = np.linalg.norm(end_to_points_vector, axis=2)

    cross_product =  start_to_points_vector[:, :, 0] * line_vector[:, 1] - start_to_points_vector[:, :, 1] * line_vector[:, 0]
    line_to_points_norm = np.abs(cross_product) / np.linalg.norm(line_vector, axis=1)
    
    dot_product = start_to_points_vector[:, :, 0] * line_vector[:, 0] + start_to_points_vector[:, :, 1] * line_vector[:, 1]
    dot_product_projection = dot_product / np.linalg.norm(line_vector, axis=1)**2
    
    mask_less_0 = dot_product_projection <= 0
    mask_more_1 = dot_product_projection >= 1
    mask_between_0_1 = np.logical_not(np.logical_or(mask_less_0, mask_more_1))
    
    print(mask_less_0, mask_more_1, mask_between_0_1)
    
    distances = np.zeros((len(points), len(line_segments) - 1))
    
    distances[mask_less_0] = start_to_points_norm[mask_less_0]
    distances[mask_more_1] = end_to_points_norm[mask_more_1]
    distances[mask_between_0_1] = line_to_points_norm[mask_between_0_1]
    
    # Check which right angle line is closer and assign it to that line segment
    
    min_values = np.min(distances, axis=1)
    is_min_value = distances == min_values[:, np.newaxis]
    print(is_min_value)
    
    min_count = np.sum(is_min_value, axis=1)
    min_occurs_more_than_once = min_count > 1
    
    label = np.zeros(min_occurs_more_than_once.shape)
    
    for i, check in enumerate(min_occurs_more_than_once):
        if check:
            label[i] = 5
        else:
            label[i] = np.argwhere(is_min_value[i])
            
    print(label)
        
    return distances, label
    
    

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

if __name__ == "__main__":
    coordinates = np.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
    coordinates = np.array([(1.6, 5.2)])
    # coordinates = np.random.rand(100,2) * 10
    line_segments = np.array([(0, 0), (1, 1), (3,4), (4, 2), (10, 10)])
    result, labels = point_line_segment_assignment(line_segments, coordinates)
    # import sys
    # sys.exit()
    # result = point_line_segment_distance(line_segments, coordinates)
    

    # labels = np.argmin(result, axis=1)
    # print(labels.shape)
    # result_ = point_line_segment_side(line_segments, coordinates)
    # labels = np.take_along_axis(result_[0], labels[:,None], axis=1).astype(int)
    # print(labels.shape)

    import matplotlib.colors
    import matplotlib.pyplot as plt
    colors = ['red','green','blue', 'purple', 'black']

    fig = plt.figure(figsize=(8,8))
    plt.plot(line_segments[..., 0], line_segments[..., 1])
    plt.scatter(coordinates[..., 0], coordinates[..., 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
    for i, txt in enumerate(np.min(result,axis=1)):
        label = labels[i]
        plt.annotate(f"{coordinates[i]} {label} {np.round(result[i], 3)}", (coordinates[i, 0], coordinates[i, 1]))
    plt.show()


