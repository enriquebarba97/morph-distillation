import pandas as pd
import numpy as np
import logging
import csv
import time

from mo_distill_utils import distill

seed = 2

# Load Pareto front data and convert to array
pareto_front = pd.read_csv('mo_pareto_fronts.csv')
pareto_front = pareto_front.to_numpy()[:, 2:]

hyperparameters = pareto_front[:, :pareto_front.shape[1] - 4]
objectives = pareto_front[:, pareto_front.shape[1] - 4:]

# Final results file
results_file = 'pareto_front_training.csv'
fieldnames = [
        "Tokenizer", "Vocab Size", "Num Hidden Layers", "Hidden Size", "Hidden Act", "Hidden Dropout Prob",
        "Intermediate Size", "Num Attention Heads", "Attention Probs Dropout Prob", "Max Sequence Length",
        "Position Embedding Type", "Learning Rate", "Batch Size", "Size", "Accuracy", "FLOPS", "Flips", "Training Time", "Evaluation Time"
]

# Iterate rows
for i in range(hyperparameters.shape[0]):
    # Get hyperparameters
    hyperparams = hyperparameters[i]
    objs = objectives[i]
    logging.info(f"Training {i} with FLOPs {objs[3]} and size {objs[0]}")
    start_time = time.time()
    accs, prediction_flips = distill([hyperparams], eval=False, surrogate=False, seed=seed, weights_file=f"pareto_{i}.bin")
    training_time = time.time()-start_time
    logging.info(f"Training took: {training_time} seconds")
    start_time = time.time()
    accs, prediction_flips = distill([hyperparams], eval=True, surrogate=False, seed=seed, weights_file=f"pareto_{i}.bin")
    evaluation_time = time.time()-start_time
    logging.info(f"Evaluation took: {evaluation_time} seconds")

    # Save final values to file
    with open(results_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # Write the header only if the file is empty
        if file.tell() == 0:
            writer.writeheader()

        # Create a dictionary from row data
        row_data = {
            "Tokenizer": hyperparams[0],
            "Vocab Size": hyperparams[1],
            "Num Hidden Layers": hyperparams[2],
            "Hidden Size": hyperparams[3],
            "Hidden Act": hyperparams[4],
            "Hidden Dropout Prob": hyperparams[5],
            "Intermediate Size": hyperparams[6],
            "Num Attention Heads": hyperparams[7],
            "Attention Probs Dropout Prob": hyperparams[8],
            "Max Sequence Length": hyperparams[9],
            "Position Embedding Type": hyperparams[10],
            "Learning Rate": hyperparams[11],
            "Batch Size": hyperparams[12],
            "Size": objs[0],  # Assuming objs[0] is the Size
            "Accuracy": accs[0],  # Assuming accs contains accuracy values
            "FLOPS": objs[3],  # Assuming objs[3] is the FLOPS
            "Flips": prediction_flips[0],  # Assuming prediction_flips contains the flips value
            "Training Time": training_time,
            "Evaluation Time": evaluation_time
        }

        # Write the row data to the CSV file
        writer.writerow(row_data)
