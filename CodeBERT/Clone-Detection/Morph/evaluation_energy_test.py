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

    eval_rounds = 5
    for i in range(eval_rounds):
        accs, prediction_flips = distill([hyperparams], eval=True, surrogate=True, seed=seed, weights_file=f"pareto_{model_number}.bin")



if __name__ == "__main__":
    main()