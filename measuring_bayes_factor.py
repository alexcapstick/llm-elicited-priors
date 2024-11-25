import os
import json
import typing as t
import tqdm
import argparse
import numpy as np
import itertools

import sklearn.metrics as skmetrics
from sklearn.utils import resample
import sklearn.preprocessing as skpre
import sklearn.impute as skimpute
import sklearn.pipeline as skpipe
from sklearn.compose import ColumnTransformer
from pathlib import Path

import pytensor.tensor as pt
import pymc as pm

from my_code.gpt import (
    get_llm_predictions,
    GPTOutputs,
)
from my_code.datasets import (
    load_fake_data,
    load_breast_cancer,
    load_uti,
    load_california_housing,
    load_heart_disease,
    load_wine_quality,
)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_repeats", type=int, default=5)
parser.add_argument("--n_samples_icl", type=int, default=1)
parser.add_argument("--n_training_points", type=int, default=25)
parser.add_argument(
    "--dataset",
    type=str,
    default=[
        "uti",
        "breast_cancer",
        "california_housing",
        "heart_disease",
        "wine_quality",
        "fake_data",
    ],
    nargs="+",
)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument(
    "--save_path",
    type=str,
    default="./results/bayes_factor",
)
args = parser.parse_args()


#### the number of features to use in the dataset.
#### this is because some of the language models have trouble predicting
#### the correct output in-context when the number of features is too large
N_FEATURES = 10

PRIORS_DIR = "./priors/elicitation"
PROMPTS_DIR = "./prompts/icl"

DATASETS = args.dataset
TEMPERATURE = args.temperature
SEED = args.seed
N_REPEATS = args.n_repeats
N_SAMPLES_ICL = args.n_samples_icl
N_TRAINING_POINTS = args.n_training_points
SAVE_PATH = args.save_path

CLIENT_CLASS = GPTOutputs
CLIENT_KWARGS = dict(
    temperature=TEMPERATURE,
    model_id="gpt-3.5-turbo-0125",
)

DATASET_FUNCTIONS = {
    "fake_data": load_fake_data,
    "breast_cancer": load_breast_cancer,
    "uti": load_uti,
    "california_housing": load_california_housing,
    "heart_disease": load_heart_disease,
    "wine_quality": load_wine_quality,
}

DATASET_MODEL_TYPES = {
    "fake_data": "linear",
    "breast_cancer": "logistic",
    "uti": "logistic",
    "california_housing": "linear",
    "heart_disease": "logistic",
    "wine_quality": "logistic",
}

DO_INCONEXT_LEARNING = {
    "fake_data": True,
    "breast_cancer": True,
    "uti": False,
    "california_housing": True,
    "heart_disease": True,
    "wine_quality": True,
}

Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)


def sample_informative_prior_predictive_logistic_regression(
    model,
    priors,
    X,
    rng,
):

    n_priors, n_features, _ = priors.shape
    with model:

        w = pm.Dirichlet("w", a=np.ones(n_priors), shape=(n_features, n_priors))

        components = pm.Normal.dist(
            mu=priors[:, :, 0].T, sigma=priors[:, :, 1].T, shape=(n_features, n_priors)
        )

        theta = pm.Mixture(
            "theta",
            w=w,
            comp_dists=components,
        )

        observations_data = pm.Data("observations", X, dims=("N", "D"))

        p = pm.Deterministic(
            "p", pm.math.invlogit(theta[0] + observations_data @ theta[1:]), dims=("N",)
        )

        # notice there is no y_true here, since we are sampling the prior predictive,
        # not the posterior predictive
        outcomes = pm.Bernoulli("outcomes", p=p, dims=("N",))

        idata = pm.sample_prior_predictive(samples=500, random_seed=rng)

        return idata.prior["p"].to_numpy().reshape(-1, X.shape[0])


def sample_uninformative_prior_predictive_logistic_regression(model, X, rng):
    with model:

        theta = pm.Normal("theta", mu=0, sigma=1, shape=X.shape[1] + 1)

        observations_data = pm.Data("observations", X, dims=("N", "D"))

        p = pm.Deterministic(
            "p", pm.math.invlogit(theta[0] + observations_data @ theta[1:]), dims=("N",)
        )

        # notice there is no y_true here, since we are sampling the prior predictive,
        # not the posterior predictive
        outcomes = pm.Bernoulli("outcomes", p=p, dims=("N",))

        idata = pm.sample_prior_predictive(samples=500, random_seed=rng)

        return idata.prior["p"].to_numpy().reshape(-1, X.shape[0])


def sample_informative_prior_predictive_linear_regression(model, priors, X, rng):

    n_priors, n_features, _ = priors.shape
    with model:

        # Define priors
        likelihood_sigma = pm.HalfCauchy(
            "sigma",
            beta=1,
        )

        w = pm.Dirichlet("w", a=np.ones(n_priors), shape=(n_features, n_priors))

        components = pm.Normal.dist(
            mu=priors[:, :, 0].T, sigma=priors[:, :, 1].T, shape=(n_features, n_priors)
        )

        theta = pm.Mixture(
            "theta",
            w=w,
            comp_dists=components,
        )

        observations_data = pm.Data("observations", X, dims=("N", "D"))

        # notice there is no y_true here, since we are sampling the prior predictive,
        # not the posterior predictive
        likelihood = pm.Normal(
            "outcomes",
            mu=theta[0] + observations_data @ theta[1:],
            sigma=likelihood_sigma,
            dims=("N",),
        )

        idata = pm.sample_prior_predictive(samples=500, random_seed=rng)

        return idata.prior["outcomes"].to_numpy().reshape(-1, X.shape[0])


def sample_uninformative_prior_predictive_linear_regression(model, X, rng):

    with model:
        likelihood_sigma = pm.HalfCauchy(
            "sigma",
            beta=1,
        )
        theta = pm.Normal(
            "theta",
            np.zeros(X.shape[1] + 1),
            sigma=1 * np.ones(X.shape[1] + 1),
        )

        observations_data = pm.Data("observations", X, dims=("N", "D"))

        # notice there is no y_true here, since we are sampling the prior predictive,
        # not the posterior predictive
        likelihood = pm.Normal(
            "outcomes",
            mu=theta[0] + observations_data @ theta[1:],
            sigma=likelihood_sigma,
            dims=("N",),
        )

        idata = pm.sample_prior_predictive(samples=500, random_seed=rng)

        return idata.prior["outcomes"].to_numpy().reshape(-1, X.shape[0])


def get_metrics_classification(
    y_true: np.ndarray,
    y_pred: t.List[np.ndarray],
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Get the log marginal likelihood and the training accuracy
    from a list of predictions from multiple runs and the
    true targets.


    Arguments
    ---------

    y_true: np.ndarray
        The true targets

    y_pred: t.List[np.ndarray]
        The list of predictions from multiple runs

    Returns
    -------

    log_marginal_likelihood: np.ndarray
        The log marginal likelihood for each run

    train_accuracy: np.ndarray
        The training accuracy for each run

    """
    log_marginal_likelihood = np.array(
        [
            -skmetrics.log_loss(
                y_true.ravel(),
                p.ravel(),
            )
            for p in y_pred
        ]
    )

    train_accuracy = np.array(
        [
            skmetrics.accuracy_score(
                y_true.ravel(),
                (p > 0.5).ravel(),
            )
            for p in y_pred
        ]
    )

    return log_marginal_likelihood, train_accuracy


def get_metrics_regression(
    y_true: np.ndarray,
    y_pred: t.List[np.ndarray],
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    Get the log marginal likelihood and the training r2
    from a list of predictions from multiple runs and the
    true targets.


    Arguments
    ---------

    y_true: np.ndarray
        The true targets

    y_pred: t.List[np.ndarray]
        The list of predictions from multiple runs

    Returns
    -------

    log_marginal_likelihood: np.ndarray
        The log marginal likelihood for each run

    train_r2: np.ndarray
        The training r2 for each run

    """

    log_marginal_likelihood = np.array(
        [
            -skmetrics.mean_squared_error(
                y_true.ravel(),
                p.ravel(),
            )
            * len(y_true.ravel())
            for p in y_pred
        ]
    )

    train_r2 = np.array(
        [
            skmetrics.r2_score(
                y_true.ravel(),
                p.ravel(),
            )
            for p in y_pred
        ]
    )

    return log_marginal_likelihood, train_r2


dataset_informative_prior_predictive_functions = {
    "fake_data": sample_informative_prior_predictive_linear_regression,
    "breast_cancer": sample_informative_prior_predictive_logistic_regression,
    "uti": sample_informative_prior_predictive_logistic_regression,
    "california_housing": sample_informative_prior_predictive_linear_regression,
    "heart_disease": sample_informative_prior_predictive_logistic_regression,
    "wine_quality": sample_informative_prior_predictive_logistic_regression,
}

dataset_uninformative_prior_predictive_functions = {
    "fake_data": sample_uninformative_prior_predictive_linear_regression,
    "breast_cancer": sample_uninformative_prior_predictive_logistic_regression,
    "uti": sample_uninformative_prior_predictive_logistic_regression,
    "california_housing": sample_uninformative_prior_predictive_linear_regression,
    "heart_disease": sample_uninformative_prior_predictive_logistic_regression,
    "wine_quality": sample_uninformative_prior_predictive_logistic_regression,
}


dataset_metrics_function = {
    "fake_data": get_metrics_regression,
    "breast_cancer": get_metrics_classification,
    "uti": get_metrics_classification,
    "california_housing": get_metrics_regression,
    "heart_disease": get_metrics_classification,
    "wine_quality": get_metrics_classification,
}


# iterating over the dataset to get the internal model
for dataset_name in tqdm.tqdm(
    DATASETS,
    total=len(DATASETS),
    desc="Iterating over datasets",
    position=0,
):

    # load priors

    priors = []

    prior_files = [f for f in os.listdir(PRIORS_DIR) if f.startswith(dataset_name)]

    for prior_file in prior_files:
        prior = np.load(os.path.join(PRIORS_DIR, prior_file))
        priors.append(prior)

    priors = np.stack(priors)

    priors = priors[:, : (N_FEATURES + 1), :]

    # getting data
    dataset = DATASET_FUNCTIONS[dataset_name](as_frame=False)
    feature_names = dataset.feature_names[:N_FEATURES]
    X, y = dataset["data"], dataset["target"]
    X, y = X[:, :N_FEATURES], y

    # load the system roles from the txt file
    system_roles = (
        open(os.path.join(PROMPTS_DIR, f"system_roles_{dataset_name}.txt"))
        .read()
        .split("\n\n")
    )

    # load the final messages from the txt file
    final_messages = (
        open(os.path.join(PROMPTS_DIR, f"final_messages_{dataset_name}.txt"))
        .read()
        .split("\n\n")
    )

    for r in range(N_REPEATS):

        # reproducibility (on our side! GPT-4 is not always deterministic)
        SEED += 1
        rng = np.random.default_rng(SEED)

        # sampling the training data
        X_train, y_train = resample(
            X,
            y,
            n_samples=N_TRAINING_POINTS,
            random_state=int(rng.integers(1e9)),
        )

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
            X_train_transform = preprocessing.fit_transform(X_train)
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
            X_train_transform = preprocessing.fit_transform(X_train)
        else:
            X_train_transform = X_train

        ## prior elicitation

        model = pm.Model()
        # sampling the prior predictive
        predictions_pe = dataset_informative_prior_predictive_functions[dataset_name](
            model=model, priors=priors, X=X_train_transform, rng=rng
        )

        # getting metrics
        log_marginal_likelihood_pe, train_metric_pe = dataset_metrics_function[
            dataset_name
        ](y_train, predictions_pe)

        # saving the prior elicitation results
        pe_save_path = os.path.join(
            SAVE_PATH,
            "elicitation",
            f"{dataset_name}_pe_{r}.npz",
        )
        os.makedirs(os.path.dirname(pe_save_path), exist_ok=True)
        np.savez(
            pe_save_path,
            predictions=predictions_pe,
            y_true=y_train,
            log_marginal_likelihood=log_marginal_likelihood_pe,
        )

        ## uninformative prior

        model = pm.Model()
        # sampling the prior predictive
        predictions_up = dataset_uninformative_prior_predictive_functions[dataset_name](
            model=model, X=X_train_transform, rng=rng
        )

        # getting metrics
        log_marginal_likelihood_up, train_metric_up = dataset_metrics_function[
            dataset_name
        ](y_train, predictions_up)

        # saving the prior elicitation results
        up_save_path = os.path.join(
            SAVE_PATH,
            "elicitation",
            f"{dataset_name}_up_{r}.npz",
        )
        os.makedirs(os.path.dirname(up_save_path), exist_ok=True)
        np.savez(
            up_save_path,
            predictions=predictions_up,
            y_true=y_train,
            log_marginal_likelihood=log_marginal_likelihood_up,
        )

        if not DO_INCONEXT_LEARNING[dataset_name]:
            continue

        ## in-context learning

        # the language model
        client = CLIENT_CLASS(**CLIENT_KWARGS)

        prompt_list = [
            (sr, fm) for sr, fm in itertools.product(system_roles, final_messages)
        ]

        predictions_icl = []
        for sr, fm in tqdm.tqdm(prompt_list, desc="Prompt", position=1):
            predictions_icl_this_prompt = []
            for s in tqdm.trange(N_SAMPLES_ICL, desc="Sample", position=2):
                predictions_icl_this_prompt.append(
                    get_llm_predictions(
                        client=client,
                        x=X_train_transform,
                        system_role=sr,
                        final_message=fm,
                        feature_names=feature_names,
                        demonstration=None,
                        dry_run=False,
                    )
                )

            # only getting the valid predictions
            predictions_icl_this_prompt = np.array(
                [
                    p
                    for p in predictions_icl_this_prompt
                    if len(p) == X_train_transform.shape[0]
                ]
            )
            if len(predictions_icl_this_prompt) != 0:
                predictions_icl.append(predictions_icl_this_prompt)

            # getting metrics
            if len(predictions_icl) != 0:
                log_marginal_likelihood_icl, train_metric_icl = (
                    dataset_metrics_function[dataset_name](y_train, predictions_icl)
                )

                # saving the prior in-context results
                icl_save_path = os.path.join(
                    SAVE_PATH,
                    "icl",
                    f"{dataset_name}_icl_{r}.npz",
                )
                os.makedirs(os.path.dirname(icl_save_path), exist_ok=True)
                np.savez(
                    icl_save_path,
                    predictions=predictions_icl,
                    y_true=y_train,
                    log_marginal_likelihood=log_marginal_likelihood_icl,
                )
