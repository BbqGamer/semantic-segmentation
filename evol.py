import random
from pathlib import Path

import numpy as np
import torch  # Import PyTorch
import yaml
from deap import algorithms, base, creator, tools

# PYTORCH_CUDA_ALLOC_CONF['expandable_segments']:True
dataset = Path("/home/konstanty/STUDIA/masters_1_sem/rob3/data_odometry_velodyne")
predictions = Path(
    "/home/konstanty/STUDIA/masters_1_sem/rob3/semantic-segmentation/predictions/"
)

with open("semantic-kitti.yaml", "r") as f:
    kitty_conf = yaml.load(f, yaml.Loader)


# --- Configuration for GPU usage ---
# Determine if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_labels(file_path):
    labels = np.fromfile(file_path, dtype=np.uint32)
    semantic_labels = labels & 0xFFFF  # mask lower 16 bits
    instance_ids = labels >> 16  # upper 16 bits
    return semantic_labels, instance_ids


def translate_element(x):
    translation_key = kitty_conf["learning_map"]
    return translation_key.get(x, x)


files = frame_files = list((predictions / "sequences/08/predictions").iterdir())
sampled_files = random.sample(files, int(len(files) * 0.1))
l = []
for f in sampled_files:
    label_file = (
        dataset
        / "dataset"
        / "sequences"
        / f.parents[1].stem
        / "labels"
        / (f.stem + ".label")
    )
    a, _ = load_labels(label_file)

    l.append(np.vectorize(translate_element)(a))
L = np.concatenate(l, axis=0)
matrices = [np.load(f) for f in sampled_files]
M = np.concatenate(matrices, axis=0)
# Convert NumPy arrays to PyTorch tensors and move them to the selected device
M_torch = torch.tensor(M, device=device, dtype=torch.float32)
L_torch = torch.tensor(L, device=device, dtype=torch.long)  # Labels should be long type

# The number of weights for the individual in the genetic algorithm
# Assuming 'w' directly corresponds to weights for each class score in M
n_weights = 19


BATCH_SIZE = 2048


# --- MIOU Fitness Function with PyTorch (GPU accelerated and Batched) ---
def fitness_mIoU_torch_batched(w):
    """
    Calculates the mean Intersection over Union (mIoU) for a given set of weights 'w'.
    This function processes the data in batches using PyTorch to manage GPU memory.

    Args:
        w (list): A list of floating-point weights representing an individual in the GA.

    Returns:
        tuple: A tuple containing the mIoU score. (e.g., (0.75,))
    """
    # Convert the individual's weights (from DEAP) to a PyTorch tensor once
    # and move it to the computational device (CPU or GPU)
    w_torch = torch.tensor(w, device=device, dtype=torch.float32)

    # Initialize dictionaries to accumulate True Positives, False Positives, and False Negatives
    # for each class across all batches. We use a dict for sparse classes or for easy iteration.
    accumulated_tp = {cls_id: 0 for cls_id in range(-1,n_weights)}
    accumulated_fp = {cls_id: 0 for cls_id in range(-1,n_weights)}
    accumulated_fn = {cls_id: 0 for cls_id in range(-1,n_weights)}

    # Iterate over the data in batches
    for i in range(0, L.shape[0], BATCH_SIZE):
        batch_M_cpu = M[i : i + BATCH_SIZE]
        batch_L_cpu = L[i : i + BATCH_SIZE]

        # Convert the current batch to PyTorch tensors and move to the device
        batch_M_torch = torch.tensor(batch_M_cpu, device=device, dtype=torch.float32)
        batch_L_torch = torch.tensor(batch_L_cpu, device=device, dtype=torch.long)

        # 1. Apply weights and get predictions for the current batch
        # Perform element-wise multiplication on the GPU
        weighted_scores_torch = batch_M_torch * w_torch

        # Get the predicted class for each sample/pixel in the batch
        predictions_torch = torch.argmax(weighted_scores_torch, dim=1)

        # Calculate TP, FP, FN for each class within the current batch
        # And accumulate them
        current_batch_unique_classes = torch.unique(batch_L_torch).tolist()

        for cls in range(-1,n_weights):  # Iterate through all possible classes
            # Convert class ID to tensor on device for comparison
            cls_torch = torch.tensor(cls, device=device, dtype=torch.long)

            # True Positives for current class in batch
            tp_batch = torch.sum(
                (predictions_torch == cls_torch) & (batch_L_torch == cls_torch)
            ).item()
            accumulated_tp[cls] += tp_batch

            # False Positives for current class in batch
            fp_batch = torch.sum(
                (predictions_torch == cls_torch) & (batch_L_torch != cls_torch)
            ).item()
            accumulated_fp[cls] += fp_batch

            # False Negatives for current class in batch
            fn_batch = torch.sum(
                (predictions_torch != cls_torch) & (batch_L_torch == cls_torch)
            ).item()
            accumulated_fn[cls] += fn_batch

    # After processing all batches, calculate mIoU using accumulated counts
    iou_per_class = []
    # Using np.unique(L_cpu) to get unique classes from the *entire* dataset
    # ensures we consider all classes relevant in the ground truth
    all_ground_truth_classes = np.unique(L)

    for cls in all_ground_truth_classes:
        tp = accumulated_tp[cls]
        fp = accumulated_fp[cls]
        fn = accumulated_fn[cls]

        denominator = tp + fp + fn

        if denominator == 0:
            # This case means the class was never truly present in L or predicted
            # If a class is not present in L at all, it's typically ignored for mIoU
            # We explicitly check the total count of the class in the ground truth
            if np.sum(L == cls) == 0:
                continue  # Skip classes not present in the overall ground truth
            else:
                # If class is present in ground truth but denominator is 0 (e.g., all FN, no TP/FP), IoU is 0.0
                iou = 0.0
        else:
            iou = tp / denominator

        iou_per_class.append(iou)

    if not iou_per_class:
        # If no valid classes were found or processed, return 0.0 mIoU
        return (0.0,)

    mIoU = np.mean(iou_per_class)
    return (mIoU,)  # DEAP fitness functions must return a tuple


def fitness_accuracy_torch_batched(individual):
    """
    Calculates the accuracy for a given set of weights 'individual'.
    This function processes the data in batches using PyTorch to manage GPU memory.

    Args:
        individual (list): A list of floating-point weights representing an individual in the GA.

    Returns:
        tuple: A tuple containing the accuracy score. (e.g., (0.92,))
    """
    # Convert the individual's weights to a PyTorch tensor and move to the device
    w_torch = torch.tensor(individual, dtype=torch.float32, device=device)

    total_correct_predictions = 0
    total_samples_processed = 0

    # Iterate over the data in batches
    for i in range(0, L.shape[0], BATCH_SIZE):
        batch_M_cpu = M[i:i + BATCH_SIZE]
        batch_L_cpu = L[i:i + BATCH_SIZE]

        # Convert the current batch to PyTorch tensors and move to the device
        batch_M_torch = torch.tensor(batch_M_cpu, device=device, dtype=torch.float32)
        batch_L_torch = torch.tensor(batch_L_cpu, device=device, dtype=torch.long)

        # Apply weights and get predicted class for the batch
        logits = batch_M_torch * w_torch
        predictions = torch.argmax(logits, dim=1)

        # Calculate correct predictions for the batch
        correct_predictions_batch = (predictions == batch_L_torch).sum().item()
        
        total_correct_predictions += correct_predictions_batch
        total_samples_processed += len(batch_L_cpu)

    if total_samples_processed == 0:
        return (0.0,) # Avoid division by zero if no samples were processed

    accuracy = total_correct_predictions / total_samples_processed
    return (accuracy,) # DEAP fitness functions must return a tuple



# --- DEAP setup (largely same as before) ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximize mIoU
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: np.random.uniform(0.01, 10.0))  # avoid 0
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_weights
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the new PyTorch-enabled fitness function
toolbox.register("evaluate", fitness_mIoU_torch_batched)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("select", tools.selTournament, tournsize=3)


# Optional: enforce positivity after mutation
def repair(individual):
    """
    Repairs an individual by ensuring all its values are positive.
    If a value is non-positive, it's replaced with a small random positive float.
    Ensures the repaired individual is of type creator.Individual.
    """
    repaired_list = list(individual)  # Create a mutable list from the individual
    for i in range(len(repaired_list)):
        if repaired_list[i] <= 0:
            repaired_list[i] = np.random.uniform(0.01, 1.0)
    return creator.Individual(repaired_list)  # Cast back to creator.Individual


def safe_mutate(individual):
    """
    Applies Gaussian mutation and then repairs the individual to ensure positivity.
    Returns a tuple containing the mutated and repaired individual.
    """
    (mutant,) = tools.mutGaussian(individual, mu=1.0, sigma=0.5, indpb=0.2)
    repaired = repair(mutant)  # repair function returns creator.Individual
    return (repaired,)  # Must return a tuple for DEAP's varAnd


toolbox.register("mutate", safe_mutate)
# --- Base case ---
base_case = (np.array([1 for _ in range(19)]))
print("Accuracy Before:", fitness_accuracy_torch_batched(base_case)[0])
print("mIoU Before:", fitness_mIoU_torch_batched(base_case)[0])
# --- Run the algorithm ---
print("\nStarting genetic algorithm...")
population = toolbox.population(n=100)  # Population size
result, log = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=0.5,
    mutpb=0.2,
    ngen=40,
    verbose=True,  # cxpb: crossover probability, mutpb: mutation probability, ngen: number of generations
)

# --- Get best individual ---
best_ind = tools.selBest(result, k=1)[0]
print("\n--- Optimization Results ---")
print("Best individual (weights):", best_ind)
# Evaluate the best individual using the PyTorch fitness function to get its mIoU
print("Accuracy:", fitness_accuracy_torch_batched(best_ind)[0])
print("mIoU:", fitness_mIoU_torch_batched(best_ind)[0])
