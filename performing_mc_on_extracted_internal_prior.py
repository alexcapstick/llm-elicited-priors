import os
import json
import numpy as np
from pathlib import Path
import tqdm
import argparse

import sklearn.pipeline as skpipe
import sklearn.preprocessing as skpre
import sklearn.impute as skimpute
from sklearn.compose import ColumnTransformer

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import blackjax

from llm_elicited_priors.datasets import (
    load_fake_data,
    load_uti,
    load_breast_cancer,
    load_california_housing,
    load_wine_quality,
    load_heart_disease,
    load_sk_diabetes,
    load_hypothyroid,
)

from llm_elicited_priors.mc import sample_posterior_from_prior_samples


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="The dataset to use for the experiments",
    nargs="+",
    default=[
        "fake_data",
        "breast_cancer",
        "heart_disease",
        "diabetes",
        "hypothyroid",
        # "wine_quality",
        # "california_housing",
    ],
)

args = parser.parse_args()

SEED = 42

# where the prior and posterior samples are stored
PRIOR_PATH = "./priors/internal_model/prior_with_multiple_messages/"
POSTERIOR_PATH = "./posteriors/internal_model/posterior_with_multiple_messages/"

# where the MC posteriors will be saved
SAVE_PATH = "posteriors/mc_on_internal_model_prior/"
Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)


for dataset in args.dataset:
    if dataset not in [
        "fake_data",
        "breast_cancer",
        "california_housing",
        "heart_disease",
        "wine_quality",
        "diabetes",
        "hypothyroid",
    ]:
        raise ValueError(f"Dataset {dataset} not recognised")

DATASETS = args.dataset

DATASET_FUNCTIONS = {
    "fake_data": load_fake_data(as_frame=False, return_X_y=False),
    "breast_cancer": load_breast_cancer(as_frame=False, return_X_y=False),
    "uti": load_uti(as_frame=False, return_X_y=False),
    "california_housing": load_california_housing(as_frame=False, return_X_y=False),
    "wine_quality": load_wine_quality(as_frame=False, return_X_y=False),
    "heart_disease": load_heart_disease(as_frame=False, return_X_y=False),
    "diabetes": load_sk_diabetes(as_frame=False, return_X_y=False),
    "hypothyroid": load_hypothyroid(as_frame=False, return_X_y=False),
}

DATASET_MODEL_TYPES = {
    "fake_data": "regression",
    "breast_cancer": "classification",
    "uti": "classification",
    "california_housing": "regression",
    "heart_disease": "classification",
    "wine_quality": "classification",
    "diabetes": "regression",
    "hypothyroid": "classification",
}

BW_METHOD = 0.25
N_CHAINS = 100
NUM_SAMPLES = 10000
NUM_ADAPT_STEPS = 1000
NUM_BURN_IN_STEPS = 300


pbar = tqdm.tqdm(total=len(DATASETS))

for dataset_name in DATASETS:

    rng = np.random.default_rng(seed=SEED)
    rng_key = jax.random.key(rng.integers(1e6))

    pbar.set_description(f"Dataset: {dataset_name}")
    pbar.refresh()

    with open(
        os.path.join(PRIOR_PATH, dataset_name, "prior_parameter_samples.json"), "r"
    ) as f:
        prior_parameter_samples = json.load(f)

    # turns the list of lists into a numpy array
    # where we remove the None values
    prior_parameter_samples = np.array(
        [b for a in prior_parameter_samples for b in a if b is not None]
    )

    total_samples = prior_parameter_samples.shape[0]
    print(f"For dataset {dataset_name}, total samples: {total_samples}")

    # loading the posterior samples
    with open(
        os.path.join(POSTERIOR_PATH, dataset_name, "posterior_parameter_samples.json"),
        "r",
    ) as f:
        posterior_parameter_samples = json.load(f)
    # loading the training indices that were used to get the posterior samples
    with open(os.path.join(POSTERIOR_PATH, dataset_name, "train_idx.json"), "r") as f:
        training_idx = json.load(f)

    n_features = prior_parameter_samples.shape[1] - 1

    for split_number in range(len(training_idx)):
        pbar.set_description(f"Dataset: {dataset_name}, split number: {split_number}")
        pbar.refresh()

        dataset = DATASET_FUNCTIONS[dataset_name]
        X, y = dataset["data"], dataset["target"]
        X = X[:, :n_features]

        X_train, y_train = X[training_idx[split_number]], y[training_idx[split_number]]

        # if we have categorical features, we dont want to scale them
        if hasattr(dataset, "categorical_features"):
            preprocessing = skpipe.Pipeline(
                [
                    (
                        "scaler",
                        ColumnTransformer(
                            transformers=[
                                (
                                    (f"{i}_no_scaling", "passthrough", [i])
                                    if i in dataset.categorical_features
                                    else (
                                        f"{i}_scaler",
                                        skpre.StandardScaler(),
                                        [i],
                                    )
                                )
                                for i in range(len(dataset.feature_names))
                            ],
                        ),
                    ),
                    (
                        "imputer",
                        skimpute.SimpleImputer(
                            strategy="mean",
                            keep_empty_features=True,
                        ),
                    ),
                ]
            )
            X_train_prep = jnp.array(preprocessing.fit_transform(X_train))
            y_train = jnp.array(y_train)
        # otherwise scale everything
        elif dataset_name != "fake_data":
            preprocessing = skpipe.Pipeline(
                [
                    ("scaler", skpre.StandardScaler()),
                    (
                        "imputer",
                        skimpute.SimpleImputer(
                            strategy="mean",
                            keep_empty_features=True,
                        ),
                    ),
                ]
            )
            X_train_prep = jnp.array(preprocessing.fit_transform(X_train))
            y_train = jnp.array(y_train)
        else:
            X_train_prep = jnp.array(X_train)
            y_train = jnp.array(y_train)

        # making the design matrix by adding the intercept as a feature of ones
        phi = jnp.concatenate(
            [
                jnp.ones(X_train_prep.shape[0])[:, None],
                X_train_prep,
            ],
            axis=-1,
        )

        # sampling from the posterior
        states = sample_posterior_from_prior_samples(
            rng_key,
            prior_parameter_samples,
            phi,
            y_train,
            algorithm=blackjax.nuts,
            classification=DATASET_MODEL_TYPES[dataset_name] == "classification",
            bw_method=BW_METHOD,
            n_chains=N_CHAINS,
            num_samples=NUM_SAMPLES,
            num_adapt_steps=NUM_ADAPT_STEPS,
        )

        # cutting out the burn-in steps
        chains = states.position[:, NUM_BURN_IN_STEPS:, :]

        # saving the posterior samples
        dataset_save_path = Path(SAVE_PATH, dataset_name)
        dataset_save_path.mkdir(parents=True, exist_ok=True)

        np.save(
            Path(
                dataset_save_path,
                f"posterior_parameter_samples_split_{split_number}",
            ),
            np.array(chains),
        )

    pbar.update()
