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
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
import argparse

from my_code.utils import load_prompts
from my_code.datasets import load_fake_data
from my_code.mc import (
    train_informative_linear_regression,
    train_uninformative_linear_regression,
    predict_model,
)
from my_code.gpt import GPTOutputs, get_llm_elicitation_for_dataset

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

POSSIBLE_MUTABLE_DESCRIPTIONS = [
    "adverserial",
    "little_information",
    "one_feature_relationship",
    "two_feature_relationship",
    "three_feature_relationship",
    "equation_feature_relationship",
]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--get_priors",
    action="store_true",
    help="Whether to get priors from the API",
)

parser.add_argument(
    "--run_mcmc",
    action="store_true",
    help="Whether to run the MCMC experiments",
)

parser.add_argument(
    "--mutable_descriptions",
    nargs="+",
    default=[],
    help="The mutable descriptions to use",
)

parser.add_argument(
    "--run_uninformative",
    action="store_true",
    help="Whether to run the MCMC experiments",
)

args = parser.parse_args()

assert all(
    md in POSSIBLE_MUTABLE_DESCRIPTIONS for md in args.mutable_descriptions
), f"Mutable descriptions must be in {POSSIBLE_MUTABLE_DESCRIPTIONS}"

PROMPTS_DIR = Path("./prompts/elicitation")
PRIORS_DIR = Path("./priors/elicitation/varied_descriptions")


##### for priors
GET_FROM_API = args.get_priors

# smallest standard deviation allowed in the priors
# as for fake data we have a std of 0.0 some times
STD_LOWER_CLIP = 1e-3

RANDOM_SEED = 2

##### for experiments
RUN_EXPERIMENTS = args.run_mcmc
RUN_UNINFORMATIVE = args.run_uninformative
N_SPLITS = 10
N_DATA_POINTS_SEEN = [2, 5, 10, 20, 30, 40, 50]
RESULTS_DIR = "./results/elicitation/varied_descriptions"

TEST_SIZE = {
    "fake_data": 0.5,
}
N_SAMPLES = 5000
N_CHAINS = 5


CLIENT_CLASS = GPTOutputs
CLIENT_KWARGS = dict(
    temperature=0.1,
    model_id="gpt-3.5-turbo-0125",
    result_args=dict(
        response_format={"type": "json_object"},
    ),
)

DATASET_FUNCTIONS = {
    "fake_data": load_fake_data(as_frame=True),
}

# ensure that the paths exist
Path(PRIORS_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


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
}

dataset_informative_training_functions = {
    "fake_data": train_informative_linear_regression,
}

dataset_split_classes = {
    "fake_data": skms.ShuffleSplit,
}

dataset_metrics_function = {
    "fake_data": get_metrics_regression,
}

dataset_metric_to_print = {
    "fake_data": "average_mse",
}

dataset_name = "fake_data"


## levels of descriptions:

adverserial = """
'target' = -5 * 'feature 0' + 2 * 'feature 1' + 3 * 'feature 2'. 
"""

little_information = """
the target is linear in features.
"""

one_feature_relationship = """
the target is a linear combination of the features
and that when 'feature 0' increases by 1, the target increases by 2.
"""

two_feature_relationship = """
the target is a linear combination of the features
and that when 'feature 0' increases by 1, the target increases by 2, and 
when 'feature 1' increases by 1, the target decreases by 1.
"""

three_feature_relationship = """
the target is a linear combination of the features
and that when 'feature 0' increases by 1, the target increases by 2, and 
when 'feature 1' increases by 1, the target decreases by 1, and when
'feature 2' increases by 1, the target increases by 1.
"""

equation_feature_relationship = """
'target' = 2 * 'feature 0' - 1 * 'feature 1' + 1 * 'feature 2'
"""


## the basline prompts

# load the system roles from the txt file
baseline_system_roles = load_prompts(
    PROMPTS_DIR.joinpath("system_roles_fake_data_varied_descriptions.txt")
)

# load the user roles from the txt file
baseline_user_roles = load_prompts(
    PROMPTS_DIR.joinpath("user_roles_fake_data_varied_descriptions.txt")
)


mutable_description_dict = {
    "adverserial": adverserial,
    "little_information": little_information,
    "one_feature_relationship": one_feature_relationship,
    "two_feature_relationship": two_feature_relationship,
    "three_feature_relationship": three_feature_relationship,
    "equation_feature_relationship": equation_feature_relationship,
}

# the ones to run
mutable_description_to_run = args.mutable_descriptions

print("Mutable descriptions to run:", mutable_description_to_run)

if GET_FROM_API:
    print(
        "Getting priors for",
        mutable_description_dict.keys(),
    )
    pbar = tqdm.tqdm(
        total=len(mutable_description_dict),
        desc="Getting priors",
        position=0,
    )

    for (
        mutable_description_name,
        mutable_description,
    ) in mutable_description_dict.items():

        if mutable_description_name not in mutable_description_to_run:
            pbar.update(1)
            continue

        system_roles = [
            sr.replace("{mutable_description}", mutable_description.replace("\n", ""))
            for sr in baseline_system_roles
        ]
        user_roles = [
            ur.replace("{mutable_description}", mutable_description.replace("\n", ""))
            for ur in baseline_user_roles
        ]

        dataset = DATASET_FUNCTIONS[dataset_name]
        client = CLIENT_CLASS(**CLIENT_KWARGS)

        priors = get_llm_elicitation_for_dataset(
            client=client,
            system_roles=system_roles,
            user_roles=user_roles,
            feature_names=dataset.feature_names.tolist(),
            target_map={k: v for v, k in enumerate(dataset.target_names)},
            verbose=True,
            std_lower_clip=STD_LOWER_CLIP,
        )

        for i, p in enumerate(priors):
            if len(p) == 0:
                print("Empty prior found, skipping")
                continue
            np.save(
                os.path.join(
                    PRIORS_DIR,
                    f"{dataset_name}_{mutable_description_name}_prior_{i}.npy",
                ),
                p,
            )

        pbar.update(1)

    pbar.close()

# we don't want to get priors for the uninformative description
# but we might want to run the experiments for it
if RUN_UNINFORMATIVE:
    mutable_description_dict["uninformative"] = None
    mutable_description_to_run.append("uninformative")


if RUN_EXPERIMENTS:
    split_rng = np.random.default_rng(RANDOM_SEED)

    dataset = DATASET_FUNCTIONS[dataset_name]

    splitter_class = dataset_split_classes[dataset_name]

    splits = splitter_class(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE[dataset_name],
        random_state=split_rng.integers(1e6),
    ).split(
        dataset.data.to_numpy(),
        dataset.target.to_numpy(),
        # splitting on groups for UTI dataset
        groups=(dataset.pid.values if dataset_name == "uti" else None),
    )

    splits = list(splits)

    pbar = tqdm.tqdm(
        total=len(mutable_description_dict) * N_SPLITS * len(N_DATA_POINTS_SEEN),
        desc="Running experiments",
    )

    for ns, (train_idx, test_idx) in enumerate(splits):
        X_train = dataset.data.iloc[train_idx].to_numpy()
        y_train = dataset.target.iloc[train_idx].to_numpy()
        X_test = dataset.data.iloc[test_idx].to_numpy()
        y_test = dataset.target.iloc[test_idx].to_numpy()

        for nps in N_DATA_POINTS_SEEN:

            X_train_seen, y_train_seen = resample(
                X_train,
                y_train,
                n_samples=nps,
                random_state=split_rng.integers(1e6),
                replace=False,
                stratify=y_train,
            )

            i = int(ns * 1e4 + nps * 1e2)

            for (
                mutable_description_name,
                mutable_description,
            ) in mutable_description_dict.items():

                experiment_rng = np.random.default_rng(RANDOM_SEED * i)

                if mutable_description_name not in mutable_description_to_run:
                    pbar.update(1)
                    i += 1
                    continue

                if mutable_description_name != "uninformative":
                    prior_files = [
                        f
                        for f in os.listdir(PRIORS_DIR)
                        if f.startswith(f"{dataset_name}_{mutable_description_name}")
                    ]
                    priors = []
                    for prior_file in prior_files:
                        prior = np.load(os.path.join(PRIORS_DIR, prior_file))
                        priors.append(prior)

                    priors = np.stack(priors)

                results_path = os.path.join(
                    RESULTS_DIR,
                    f"prior_no_prior_results_{dataset_name}_{mutable_description_name}_{ns}.json",
                )

                print(
                    f"Running experiments for {dataset_name}, {mutable_description_name}, {ns}"
                )
                print("Results will be saved to:", results_path)

                if Path(results_path).exists():
                    with open(results_path, "r") as f:
                        results = json.load(f)
                        print(
                            f"Loaded results for {dataset_name}, {mutable_description_name}, {ns}"
                        )
                else:
                    results = {}

                if dataset_name not in results:
                    results[dataset_name] = {}

                if mutable_description_name not in results[dataset_name]:
                    results[dataset_name][mutable_description_name] = {}

                # make ns int instead of str
                if len(results[dataset_name][mutable_description_name]) > 0:
                    results[dataset_name][mutable_description_name] = {
                        int(k): v
                        for k, v in results[dataset_name][
                            mutable_description_name
                        ].items()
                    }

                if ns not in results[dataset_name][mutable_description_name]:
                    results[dataset_name][mutable_description_name][ns] = {}

                # make ns int instead of str
                if len(results[dataset_name][mutable_description_name][ns]) > 0:
                    results[dataset_name][mutable_description_name][ns] = {
                        int(k): v
                        for k, v in results[dataset_name][mutable_description_name][
                            ns
                        ].items()
                    }

                if nps not in results[dataset_name][mutable_description_name][ns]:
                    results[dataset_name][mutable_description_name][ns][nps] = {}
                else:
                    if (
                        "informative"
                        in results[dataset_name][mutable_description_name][ns][nps]
                    ):
                        pbar.update(1)
                        print(f"Skipping {dataset_name}, {ns}, {nps}")
                        i += 1
                        continue

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

                model = pm.Model()

                if mutable_description_name == "uninformative":
                    training_function = dataset_uninformative_training_functions[
                        dataset_name
                    ]
                    args = dict()
                else:
                    training_function = dataset_informative_training_functions[
                        dataset_name
                    ]
                    args = dict(priors=priors)

                idata_informative, model = training_function(
                    model,
                    X_train=X_train_seen_transformed,
                    y_train=y_train_seen,
                    rng=experiment_rng,
                    n_samples=N_SAMPLES,
                    n_chains=N_CHAINS,
                    **args,
                )

                posterior_informative = predict_model(
                    model,
                    idata_informative,
                    X_test_transformed,
                    experiment_rng,
                )

                y_pred_test_informative = posterior_informative["predictions"][
                    "outcomes"
                ].to_numpy()

                results[dataset_name][mutable_description_name][ns][nps][
                    "informative"
                ] = dataset_metrics_function[dataset_name](
                    y_test, y_pred_test_informative
                )
                pbar.update(1)

                metric_to_print = dataset_metric_to_print[dataset_name]

                value_to_print = results[dataset_name][mutable_description_name][ns][
                    nps
                ]["informative"][metric_to_print]

                print(
                    "\n",
                    f"Dataset: {dataset_name}, Split: {ns+1}, N Points Seen: {nps},",
                    "\n",
                    f"Informative {metric_to_print}:",
                    f"{value_to_print}",
                    "\n",
                    "-" * 80,
                    "\n",
                )
                i += 1

                print("Saving results to:", results_path)
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=4)

    pbar.close()
