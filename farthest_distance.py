import pandas as pd

def find_farthest_distance(coordinates):

    data = coordinates

    # Group the data by the shape index (column 'f')
    shape_groups = data.groupby('f')

    # Store the coordinates for each shape in 2D (x, y)
    shapes = []
    for shape_idx, group in shape_groups:
        coordinates = group[['x', 'y']].values
        shapes.append(coordinates)


    # Function to calculate the maximum x and y distances between the farthest points of two shapes
    def farthest_point_distance(shape1, shape2):
        max_x_distance = 0
        max_y_distance = 0

        for p1 in shape1:
            for p2 in shape2:
                x_distance = abs(p1[0] - p2[0])  # Difference in x-coordinates
                y_distance = abs(p1[1] - p2[1])  # Difference in y-coordinates

                if x_distance > max_x_distance:
                    max_x_distance = x_distance
                if y_distance > max_y_distance:
                    max_y_distance = y_distance

        return int(max_x_distance), int(max_y_distance)  # Return as plain integers


    # Calculate the farthest distances in x and y directions between each pair of shapes
    farthest_distances = {}
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            max_x, max_y = farthest_point_distance(shapes[i], shapes[j])
            farthest_distances[f"F_{i} and F_{j}"] = (max_x, max_y)

    # Output the farthest distances
    return farthest_distances
