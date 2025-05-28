import os
from pathlib import Path
import json
import tqdm
import numpy as np
import pymc as pm
import sklearn.model_selection as skms
import sklearn.preprocessing as skpre
import sklearn.impute as skimpute
import sklearn.pipeline as skpipe
import sklearn.metrics as skmetrics
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
import argparse
import itertools

from llm_elicited_priors.datasets import (
    load_fake_data,
    load_uti,
    load_breast_cancer,
    load_california_housing,
    load_heart_disease,
    load_wine_quality,
    load_sk_diabetes,
    load_hypothyroid,
)
from llm_elicited_priors.utils import load_prompts
from llm_elicited_priors.mc import (
    train_informative_linear_regression,
    train_uninformative_linear_regression,
    train_informative_logistic_regression,
    train_uninformative_logistic_regression,
    predict_model,
)

POSSIBLE_DATASETS = [
    "fake_data",
    "uti",
    "breast_cancer",
    "heart_disease",
    "diabetes",
    "hypothyroid",
    # "california_housing",
    # "wine_quality",
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="The dataset(s) to use for the experiments",
    nargs="+",
    default=POSSIBLE_DATASETS,
)

parser.add_argument(
    "--model",
    type=str,
    help="The model to use for the experiments",
    default="gpt-3.5-turbo-0125",
)


parser.add_argument(
    "--quantisation",
    type=str,
    help="The quantisation to use for the experiments. "
    "This only applies to the llama models and can be one "
    "of 'none', 'bfloat16', 'int8', 'int4'",
    default="none",
)

PROMPTS_DIR = "./prompts/elicitation"
PRIORS_DIR = "./priors/elicitation"
N_REPEATS = 10
RANDOM_SEED = 42


N_PRIORS_TO_USE_LIST = [4, 6, 8, 10]


args = parser.parse_args()

for dataset in args.dataset:
    if dataset not in POSSIBLE_DATASETS:
        raise ValueError(f"Dataset {dataset} not recognised")

datasets_to_test = args.dataset

##### for experiments
N_SPLITS = 10
N_DATA_POINTS_SEEN = [5, 10, 20, 30, 40, 50]
RESULTS_DIR = "./results/elicitation_varied_numbers_of_descriptions"


DATASETS_TO_EXPERIMENT = args.dataset
TEST_SIZE = {
    "fake_data": 0.5,
    "uti": 0.5,
    "breast_cancer": 0.5,
    "california_housing": 0.5,
    "heart_disease": 0.5,
    "wine_quality": 0.5,
    "diabetes": 0.5,
    "hypothyroid": 0.5,
}
N_SAMPLES = 5000
N_CHAINS = 5

POSSIBLE_MODELS = [
    "uninformative",
    "gpt-3.5-turbo-0125",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo-2024-04-09",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "deepseek-r1:32b",
    "deepseek-r1:14b-qwen-distill-fp16",
]

if args.model not in POSSIBLE_MODELS:
    raise ValueError(f"Model {args.model} not recognised")


if args.quantisation != "none":
    args.model += f"-{args.quantisation}"

print("Using model:", args.model)

# we want to save the priors and results in a subfolder
PRIORS_DIR = os.path.join(
    PRIORS_DIR,
    args.model.replace("/", "-").replace(".", "-").replace(":", "-").lower(),
)
Path(PRIORS_DIR).mkdir(parents=True, exist_ok=True)

RESULTS_DIR = os.path.join(
    RESULTS_DIR, 
    args.model.replace("/", "-").replace(".", "-").replace(":", "-").lower(),
)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


DATASET_FUNCTIONS = {
    "fake_data": load_fake_data,
    "breast_cancer": load_breast_cancer,
    "uti": load_uti,
    "california_housing": load_california_housing,
    "heart_disease": load_heart_disease,
    "wine_quality": load_wine_quality,
    "diabetes": load_sk_diabetes,
    "hypothyroid": load_hypothyroid,
}


print("using the directory for priors:", PRIORS_DIR)
print("using the directory for results:", RESULTS_DIR)


def calculate_accuracies(y_pred, y_true):
    return (y_pred == y_true.reshape(1, 1, -1)).mean(-1).tolist()


def get_metrics_classification(y_true, y_pred):
    accuracy = calculate_accuracies(y_pred, y_true)
    average_prediction = np.mean(y_pred.reshape(-1, y_pred.shape[-1]), axis=0)
    roc_auc = skmetrics.roc_auc_score(y_true, average_prediction)
    metrics_dict = {
        "accuracy": accuracy,
        "average_accuracy": np.mean(accuracy),
        "average_auc": roc_auc,
    }

    return metrics_dict


def calculate_mse(y_pred, y_true):
    return ((y_pred - y_true.reshape(1, 1, -1)) ** 2).mean(-1).tolist()


def get_metrics_regression(y_true, y_pred):
    mse = calculate_mse(y_pred, y_true)
    metrics_dict = {
        "mse": mse,
        "average_mse": np.mean(mse),
    }

    return metrics_dict


dataset_uninformative_training_functions = {
    "fake_data": train_uninformative_linear_regression,
    "breast_cancer": train_uninformative_logistic_regression,
    "uti": train_uninformative_logistic_regression,
    "california_housing": train_uninformative_linear_regression,
    "heart_disease": train_uninformative_logistic_regression,
    "wine_quality": train_uninformative_logistic_regression,
    "diabetes": train_uninformative_linear_regression,
    "hypothyroid": train_uninformative_logistic_regression,
}

dataset_informative_training_functions = {
    "fake_data": train_informative_linear_regression,
    "breast_cancer": train_informative_logistic_regression,
    "uti": train_informative_logistic_regression,
    "california_housing": train_informative_linear_regression,
    "heart_disease": train_informative_logistic_regression,
    "wine_quality": train_informative_logistic_regression,
    "diabetes": train_informative_linear_regression,
    "hypothyroid": train_informative_logistic_regression,
}

dataset_split_classes = {
    "fake_data": skms.ShuffleSplit,
    "breast_cancer": skms.StratifiedShuffleSplit,
    "uti": skms.GroupShuffleSplit,
    "california_housing": skms.ShuffleSplit,
    "heart_disease": skms.StratifiedShuffleSplit,
    "wine_quality": skms.StratifiedShuffleSplit,
    "diabetes": skms.ShuffleSplit,
    "hypothyroid": skms.StratifiedShuffleSplit,
}

dataset_metrics_function = {
    "fake_data": get_metrics_regression,
    "breast_cancer": get_metrics_classification,
    "uti": get_metrics_classification,
    "california_housing": get_metrics_regression,
    "heart_disease": get_metrics_classification,
    "wine_quality": get_metrics_classification,
    "diabetes": get_metrics_regression,
    "hypothyroid": get_metrics_classification,
}

dataset_metric_to_print = {
    "fake_data": "average_mse",
    "breast_cancer": "average_accuracy",
    "uti": "average_accuracy",
    "california_housing": "average_mse",
    "heart_disease": "average_accuracy",
    "wine_quality": "average_accuracy",
    "diabetes": "average_mse",
    "hypothyroid": "average_accuracy",
}


pbar = tqdm.tqdm(
    total=len(DATASETS_TO_EXPERIMENT)
    * N_SPLITS
    * len(N_DATA_POINTS_SEEN)
    * len(N_PRIORS_TO_USE_LIST),
    desc="Running experiments",
)

for dataset_name in DATASETS_TO_EXPERIMENT:
    dataset = DATASET_FUNCTIONS[dataset_name](as_frame=True)

    rng = np.random.default_rng(RANDOM_SEED)

    splitter_class = dataset_split_classes[dataset_name]

    splits = splitter_class(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE[dataset_name],
        random_state=rng.integers(1e6),
    ).split(
        dataset.data.to_numpy(),
        dataset.target.to_numpy(),
        # splitting on groups for UTI dataset
        groups=(dataset.pid.values if dataset_name == "uti" else None),
    )

    prior_files = [f for f in os.listdir(PRIORS_DIR) if f.startswith(dataset_name)]
    priors = []
    for prior_file in prior_files:
        prior = np.load(os.path.join(PRIORS_DIR, prior_file))
        priors.append(prior)

    priors = np.stack(priors)

    splits = list(splits)

    i = 1

    for n_priors_to_use in N_PRIORS_TO_USE_LIST:

        for ns, (train_idx, test_idx) in enumerate(splits):

            rng1 = np.random.default_rng(RANDOM_SEED * i)

            # load the system roles from the txt file
            system_roles = load_prompts(
                os.path.join(PROMPTS_DIR, f"system_roles_{dataset_name}.txt")
            )

            # load the user roles from the txt file
            user_roles = load_prompts(
                os.path.join(PROMPTS_DIR, f"user_roles_{dataset_name}.txt")
            )

            # get the idx of the prompts being included:
            system_role_idx = rng1.choice(
                len(system_roles), size=n_priors_to_use, replace=False
            )
            user_role_idx = rng1.choice(
                len(user_roles), size=n_priors_to_use, replace=False
            )

            idx_to_use = list(itertools.product(system_role_idx, user_role_idx))
            idx_to_use = [x1 * len(user_roles) + x2 for x1, x2 in idx_to_use]

            priors_to_use = priors[idx_to_use]

            results_path = os.path.join(
                RESULTS_DIR,
                f"prior_no_prior_results_{dataset_name}_{n_priors_to_use}_{ns}.json",
            )

            print(
                f"Running experiments for "
                f"dataset {dataset_name}, n_priors_to_use {n_priors_to_use}, ns {ns}"
            )
            print("Results will be saved to:", results_path)

            if Path(results_path).exists():
                with open(results_path, "r") as f:
                    results = json.load(f)
                    print(
                        f"Loaded results for dataset "
                        f"{dataset_name}, n_priors_to_use {n_priors_to_use}, ns {ns}"
                    )
            else:
                results = {}

            if dataset_name not in results:
                results[dataset_name] = {}

            # make n_priors_to_use int instead of str
            if len(results[dataset_name]) > 0:
                results[dataset_name] = {
                    int(k): v for k, v in results[dataset_name].items()
                }

            if n_priors_to_use not in results[dataset_name]:
                results[dataset_name][n_priors_to_use] = {}

            # make ns int instead of str
            if len(results[dataset_name][n_priors_to_use]) > 0:
                results[dataset_name][n_priors_to_use] = {
                    int(k): v for k, v in results[dataset_name][n_priors_to_use].items()
                }

            if ns not in results[dataset_name][n_priors_to_use]:
                results[dataset_name][n_priors_to_use][ns] = {}

            X_train = dataset.data.iloc[train_idx].to_numpy()
            y_train = dataset.target.iloc[train_idx].to_numpy()
            X_test = dataset.data.iloc[test_idx].to_numpy()
            y_test = dataset.target.iloc[test_idx].to_numpy()

            for nps in N_DATA_POINTS_SEEN:

                rng2 = np.random.default_rng(RANDOM_SEED * i)

                # make nps int instead of str
                if len(results[dataset_name][n_priors_to_use][ns]) > 0:
                    results[dataset_name][n_priors_to_use][ns] = {
                        int(k): v
                        for k, v in results[dataset_name][n_priors_to_use][ns].items()
                    }

                # results[dataset_name][ns][npr][nps] = {}
                if nps not in results[dataset_name][n_priors_to_use][ns]:
                    results[dataset_name][n_priors_to_use][ns][nps] = {}
                else:
                    if "informative" in results[dataset_name][n_priors_to_use][ns][nps]:
                        pbar.update(1)
                        i += 1
                        print(
                            f"Skipping dataset {dataset_name}, "
                            f"n_priors_to_use {n_priors_to_use}, ns {ns}, "
                            f"number_of_points {nps}"
                        )
                        continue

                X_train_seen, y_train_seen = resample(
                    X_train,
                    y_train,
                    n_samples=nps,
                    random_state=rng2.integers(1e6),
                    replace=False,
                    stratify=y_train,
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
                    X_train_seen_transformed = preprocessing.fit_transform(X_train_seen)
                    X_test_transformed = preprocessing.transform(X_test)
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
                    X_train_seen_transformed = preprocessing.fit_transform(X_train_seen)
                    X_test_transformed = preprocessing.transform(X_test)
                else:
                    X_train_seen_transformed = X_train_seen
                    X_test_transformed = X_test

                model_informative = pm.Model()

                informative_training_function = dataset_informative_training_functions[
                    dataset_name
                ]

                print(
                    f"Training informative model for dataset {dataset_name}, "
                    f"n_priors_to_use {n_priors_to_use}, ns {ns}, number_of_points {nps}"
                )
                print("shape of priors:", priors_to_use.shape)

                idata_informative, model_informative = informative_training_function(
                    model_informative,
                    priors=priors_to_use,
                    X_train=X_train_seen_transformed,
                    y_train=y_train_seen,
                    rng=rng2,
                    n_samples=N_SAMPLES,
                    n_chains=N_CHAINS,
                )

                posterior_informative = predict_model(
                    model_informative, idata_informative, X_test_transformed, rng2
                )

                y_pred_test_informative = posterior_informative["predictions"][
                    "outcomes"
                ].to_numpy()

                results[dataset_name][n_priors_to_use][ns][nps]["informative"] = (
                    dataset_metrics_function[dataset_name](
                        y_test, y_pred_test_informative
                    )
                )
                pbar.update(1)

                metric_to_print = dataset_metric_to_print[dataset_name]

                print(
                    "\n",
                    f"Dataset: {dataset_name}, Split: {ns+1}, N Points Seen: {nps},",
                    "\n",
                    f"Informative {metric_to_print}:",
                    f"{results[dataset_name][n_priors_to_use][ns][nps]['informative'][metric_to_print]}",
                    "\n",
                    "-" * 80,
                    "\n",
                )

                i += 1

                print("Saving results to:", results_path)
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)

pbar.close()
