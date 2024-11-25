import os
import tqdm
import json
import itertools
from pathlib import Path
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.stats as stats

from my_code.metrics import energy_distance_chunked
from my_code.utils import load_prompts

PROMPTS_DIR = "./prompts/elicitation"
PRIORS_DIR = "./priors/elicitation"
SAVE_DIR = "./results/elicitation"
N_REPEATS = 10
SEED = 42

dataset_order = [
    "fake_data",
    "uti",
    "breast_cancer",
    "california_housing",
    "wine_quality",
    "heart_disease",
]

results = {}

n_priors_to_use_list = [4, 5, 6, 7, 8, 9, 10]
for dataset_name in tqdm.tqdm(dataset_order, desc="Datasets", position=0):
    RNG = np.random.default_rng(SEED)

    # load all priors
    priors = []

    prior_files = [f for f in os.listdir(PRIORS_DIR) if f.startswith(dataset_name)]

    for prior_file in prior_files:
        prior = np.load(os.path.join(PRIORS_DIR, prior_file))
        priors.append(prior)

    priors = np.stack(priors)[:, 1:, :]  # not using the intercept

    print("shape of priors:", priors.shape)
    n_priors, n_features, _ = priors.shape
    with pm.Model() as model:

        w = pm.Dirichlet("w", a=np.ones(n_priors), shape=(n_features, n_priors))

        components = pm.Normal.dist(
            mu=priors[:, :, 0].T, sigma=priors[:, :, 1].T, shape=(n_features, n_priors)
        )

        theta = pm.Mixture(
            "theta",
            w=w,
            comp_dists=components,
        )

        prior_samples_pe = pm.sample_prior_predictive(samples=10000)

    prior_samples_pe = prior_samples_pe["prior"]["theta"].to_numpy().squeeze()

    results[dataset_name] = {}

    for repeat in tqdm.tqdm(range(N_REPEATS), desc="Repeats", position=1):

        results[dataset_name][repeat] = {}

        for n_priors_to_use in tqdm.tqdm(
            n_priors_to_use_list, desc="Priors", position=2
        ):

            # load the system roles from the txt file
            system_roles = load_prompts(
                os.path.join(PROMPTS_DIR, f"system_roles_{dataset_name}.txt")
            )

            # load the user roles from the txt file
            user_roles = load_prompts(
                os.path.join(PROMPTS_DIR, f"user_roles_{dataset_name}.txt")
            )

            # get the idx of the prompts being included:
            system_role_idx = RNG.choice(
                len(system_roles), size=n_priors_to_use, replace=False
            )
            user_role_idx = RNG.choice(
                len(user_roles), size=n_priors_to_use, replace=False
            )

            idx_to_use = list(itertools.product(system_role_idx, user_role_idx))
            idx_to_use = [x1 * len(user_roles) + x2 for x1, x2 in idx_to_use]

            priors_to_use = priors[idx_to_use]

            n_priors, n_features, _ = priors_to_use.shape
            with pm.Model() as model:

                w = pm.Dirichlet("w", a=np.ones(n_priors), shape=(n_features, n_priors))

                components = pm.Normal.dist(
                    mu=priors_to_use[:, :, 0].T,
                    sigma=priors_to_use[:, :, 1].T,
                    shape=(n_features, n_priors),
                )

                theta = pm.Mixture(
                    "theta",
                    w=w,
                    comp_dists=components,
                )

                prior_subset_samples_pe = pm.sample_prior_predictive(samples=10000)

            prior_subset_samples_pe = (
                prior_subset_samples_pe["prior"]["theta"].to_numpy().squeeze()
            )

            # calculate the energy distance between the two sets of samples
            ed = energy_distance_chunked(
                prior_samples_pe, prior_subset_samples_pe, chunk_size=1000
            )

            results[dataset_name][repeat][n_priors_to_use] = ed.item()

    save_file = Path(SAVE_DIR).joinpath("fewer_descriptions.json")

    with open(save_file, "w") as f:
        json.dump(results, f)
