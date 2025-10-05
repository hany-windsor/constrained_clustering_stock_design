from ortools.sat.python import cp_model
import hausdorff
import farthest_distance
import pandas as pd
import time  # Import the time module

# Create the model
model = cp_model.CpModel()

# Load the CSV file
file_path = 'case_1.csv'
data = pd.read_csv(file_path)

# Extract unique feature identifiers from the 'f' column
unique_features = data['f'].unique().tolist()

# Create a dynamic index mapping using formatted strings (e.g., 'F_0')
index_mapping = {f'F_{feature}': idx for idx, feature in enumerate(unique_features)}

# Calculate Hausdorff distances
hausdorff_distances = hausdorff.find_hausdorff_distances(data)

# Initialize an empty matrix for Hausdorff distances based on the number of features
num_of_features = len(unique_features)
h = [[0] * num_of_features for _ in range(num_of_features)]

# Populate the h_ij matrix with Hausdorff distances from the dictionary
for key, value in hausdorff_distances.items():
    shape1, shape2 = key.split(' and ')

    if shape1 in index_mapping and shape2 in index_mapping:
        i, j = index_mapping[shape1], index_mapping[shape2]
    else:
        raise KeyError(f"Shape {shape1} or {shape2} not found in index mapping!")

    h[i][j] = value


# Calculate farthest x and y distances
farthest_x_y_distances = farthest_distance.find_farthest_distance(data)

# Initialize matrices for w (x distances) and l (y distances)
w = [[0] * num_of_features for _ in range(num_of_features)]
l = [[0] * num_of_features for _ in range(num_of_features)]

# Populate w and l matrices with x and y distances from the dictionary
for key, value in farthest_x_y_distances.items():
    shape1, shape2 = key.split(' and ')

    if shape1 in index_mapping and shape2 in index_mapping:
        i, j = index_mapping[shape1], index_mapping[shape2]
    else:
        raise KeyError(f"Shape {shape1} or {shape2} not found in index mapping!")

    w[i][j], l[i][j] = value[0], value[1]
    w[j][i], l[j][i] = value[0], value[1]  # Ensure symmetry

# Parameters
M = num_of_features  # Number of features
N = 5  # Number of clusters

# Binary variables: a_ik and a_jk
a = {(i, k): model.NewBoolVar(f'a_{i}_{k}') for i in range(M) for k in range(N)}

# Auxiliary variables for the product p_ijk = a_ik * a_jk
p = {}
for i in range(M):
    for j in range(M):
        for k in range(N):
            if i != j:
                p[(i, j, k)] = model.NewBoolVar(f'product_{i}_{j}_{k}')
                # Linearization of the product a_ik * a_jk
                model.Add(p[(i, j, k)] <= a[(i, k)])
                model.Add(p[(i, j, k)] <= a[(j, k)])
                model.Add(a[(i, k)] + a[(j, k)] - 1 <= p[(i, j, k)])

# Objective: Minimize z = sum(p_ijk * h_ij)
model.Minimize(
    sum(p[(i, j, k)] * h[i][j] for i in range(M) for j in range(M) for k in range(N) if i != j)
)

# Constraints for width and length
W, L = 100, 100 # Width and Length constraints

for i in range(M):
    for j in range(M):
        for k in range(N):
            if i != j:
                model.Add(p[(i, j, k)] * w[i][j] <= W)
                model.Add(p[(i, j, k)] * l[i][j] <= L)

# Assignment constraints: Each feature is assigned to exactly one cluster
for i in range(M):
    model.Add(sum(a[(i, k)] for k in range(N)) == 1)

# Ensure that each cluster has at least one feature assigned
for k in range(N):
    model.Add(sum(a[(i, k)] for i in range(M)) >= 1)

# Print the number of variables and constraints
print(f"\nNumber of variables: {len(model.Proto().variables)}")
print(f"Number of constraints: {len(model.Proto().constraints)}")

# Measure the start time
start_time = time.time()

# Solver configuration with a 2-hour time limit (7200 seconds)
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 7200  # Set time limit to 2 hours

# Solve the model
status = solver.Solve(model)

# Measure the end time
end_time = time.time()
computation_time = end_time - start_time

# Prepare the results table
results = []

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):

    print("\nDetailed Variable Assignments:")
    for i in range(M):
        for k in range(N):
            if solver.Value(a[(i, k)]) == 1:
                results.append({'Feature': f'F_{unique_features[i]}', 'Cluster': k})

    # Convert results to a DataFrame for table display
    results_df = pd.DataFrame(results)

    # Print the results as a table
    print("\nClustering Results (Table Format):")
    print(results_df.to_string(index=False))

    # Print the computational time
    print(f"\nComputational Time: {computation_time:.4f} seconds")

    # Print the objective function value
    print(f"\nObjective Function Value: {solver.ObjectiveValue():.4f}")

    # Initialize dictionary for cluster assignments
    cluster_assignments = {k: [] for k in range(N)}

    # Assign features to clusters
    for i in range(M):
        for k in range(N):
            if solver.Value(a[(i, k)]) == 1:
                # Remove 'F_' and add 1 to the feature index
                feature_number = int(unique_features[i]) + 1
                cluster_assignments[k].append(str(feature_number))

    # Print clustering results
    print("\nCluster Assignments:")
    for cluster, features in cluster_assignments.items():
        print(f"Cluster {cluster}: {', '.join(features)}")

    # Save the results in a variable (for later use if needed)
    results_dict = cluster_assignments

else:
    print('No solution found.')

# Check if time limit was reached
if status == cp_model.INFEASIBLE:
    print("Model is infeasible.")
elif status == cp_model.MODEL_INVALID:
    print("Model is invalid.")
elif status == cp_model.UNKNOWN:
    print("Solver stopped before finding an optimal or feasible solution.")


