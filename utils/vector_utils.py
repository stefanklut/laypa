import numpy as np

def point_line_segment_distance(line_segments, points):
    line_ends = line_segments[1:]
    line_starts = line_segments[:-1]

    line_dirs = line_ends - line_starts
    start_to_points = points[:, np.newaxis, :] - line_starts
    start_to_points_norm = np.linalg.norm(start_to_points, axis=2)
    end_to_points = points[:, np.newaxis, :] - line_ends
    end_to_points_norm = np.linalg.norm(end_to_points, axis=2)

    cross_product =  start_to_points[:, :, 0] * line_dirs[:, 1] - start_to_points[:, :, 1] * line_dirs[:, 0]
    line_to_points_norm = np.abs(cross_product) / np.linalg.norm(line_dirs, axis=1)
    
    dot_product = start_to_points[:, :, 0] * line_dirs[:, 0] + start_to_points[:, :, 1] * line_dirs[:, 1]
    dot_product_projection = dot_product / np.linalg.norm(line_dirs, axis=1)**2
    
    mask_less_0 = dot_product_projection <= 0
    mask_more_1 = dot_product_projection >= 1
    print(dot_product)
    print(np.linalg.norm(line_dirs, axis=1))
    print(dot_product_projection)
    mask_between_0_1 = np.logical_not(np.logical_or(mask_less_0, mask_more_1))
    print(mask_less_0)
    print(mask_more_1)
    print(mask_between_0_1)
    
    print(start_to_points_norm)
    print(end_to_points_norm)
    print(line_to_points_norm)
    
    distances = np.zeros((len(points), len(line_segments) - 1))
    
    distances[mask_less_0] = start_to_points_norm[mask_less_0]
    distances[mask_more_1] = end_to_points_norm[mask_more_1]
    
    distances[mask_between_0_1] = line_to_points_norm[mask_between_0_1]
    
    return distances

# coordinates = np.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
coordinates = np.random.rand(1000,2) * 10
line_segments = np.array([(0, 0), (1, 1), (3,8), (4, 2), (10, 10)])
result = point_line_segment_distance(line_segments, coordinates)

print(result)

labels = np.argmin(result, axis=1)

print()

import matplotlib.colors
import matplotlib.pyplot as plt
colors = ['red','green','blue', 'purple']

fig = plt.figure(figsize=(8,8))
plt.plot(line_segments[..., 0], line_segments[..., 1])
plt.scatter(coordinates[..., 0], coordinates[..., 1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
# for i, txt in enumerate(np.min(result,axis=1)):
#     label = labels[i]
#     plt.annotate(f"{coordinates[i]} {label} {np.round(result[i], 3)}", (coordinates[i, 0], coordinates[i, 1]))
plt.show()


