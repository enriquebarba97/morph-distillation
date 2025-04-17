import pandas as pd
import numpy as np
import logging
import csv
import argparse
import time

from mo_distill_utils import distill

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MORPH models.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to test energy performance.")
    args = parser.parse_args()

    model_number = int(args.model)

    seed = 2

    # Load Pareto front data and convert to array
    pareto_front = pd.read_csv('mo_pareto_fronts.csv')
    pareto_front = pareto_front.to_numpy()[:, 2:]

    hyperparameters = pareto_front[:, :pareto_front.shape[1] - 4]
    objectives = pareto_front[:, pareto_front.shape[1] - 4:]

    hyperparams = hyperparameters[model_number]
    objs = objectives[model_number]
    logging.info(f"Training {model_number} with FLOPs {objs[3]} and size {objs[0]}")
    start_time = time.time()
    accs, prediction_flips = distill([hyperparams], eval=False, surrogate=True, seed=seed)
    training_time = time.time()-start_time
    logging.info(f"Training took: {training_time} seconds")


if __name__ == "__main__":
    main()