import pandas as pd
from scipy.spatial.distance import directed_hausdorff

def find_hausdorff_distances(coordinates):

    data = coordinates

    # Group the data by the shape index (column 'f')
    shape_groups = data.groupby('f')

    # Store the coordinates for each shape in 3D (x, y, z)
    shapes = []
    for shape_idx, group in shape_groups:
        coordinates = group[['x', 'y', 'z']].values
        shapes.append(coordinates)

    # Function to calculate the Hausdorff distance between two 3D shapes
    def hausdorff_distance(shape1, shape2):
        return max(directed_hausdorff(shape1, shape2)[0], directed_hausdorff(shape2, shape1)[0])

    # Calculate the Hausdorff distances between the 3D shapes
    hausdorff_distances = {}
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            dist = hausdorff_distance(shapes[i], shapes[j])
            hausdorff_distances[f"F_{i} and F_{j}"] = dist

    # Output the Hausdorff distances
    return hausdorff_distances

