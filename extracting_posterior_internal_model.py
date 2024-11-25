import os
import json
import typing as t
import tqdm
import argparse
import numpy as np
import itertools
import sklearn.preprocessing as skpre
import sklearn.impute as skimpute
import sklearn.pipeline as skpipe
from sklearn.compose import ColumnTransformer
import sklearn.model_selection as skms
from pathlib import Path

from my_code.utils import make_list
from my_code.datasets import (
    load_fake_data,
    load_breast_cancer,
    load_california_housing,
    load_heart_disease,
    load_wine_quality,
)
from my_code.gpt import (
    sample_approximate_llm_internal_predictive_model_parameters,
    GPTOutputs,
)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_samples", type=int, default=5)
parser.add_argument("--n_datapoints_in_sample", type=int, default=25)
parser.add_argument("--n_repeats", type=int, default=5)
parser.add_argument("--n_training_points", type=int, default=25)
parser.add_argument(
    "--dataset",
    type=str,
    default=[
        "breast_cancer",
        "california_housing",
        "heart_disease",
        "wine_quality",
        "fake_data",
    ],
    nargs="+",
)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--x_sample_low", type=float, default=-5)
parser.add_argument("--x_sample_high", type=float, default=5)
parser.add_argument(
    "--save_path",
    type=str,
    default="./posteriors/internal_model/posterior_with_multiple_messages/",
)
args = parser.parse_args()

#### the number of features to use in the dataset.
#### this is because some of the language models have trouble predicting
#### the correct output in-context when the number of features is too large
N_FEATURES = 10

N_REPEATS = args.n_repeats
N_TRAINING_POINTS = args.n_training_points

DATASETS = args.dataset
TEMPERATURE = args.temperature
SEED = args.seed
N_SAMPLES = args.n_samples
N_DATAPOINTS_IN_SAMPLE = args.n_datapoints_in_sample
X_SAMPLE_LOW = args.x_sample_low
X_SAMPLE_HIGH = args.x_sample_high
SAVE_PATH = args.save_path


CLIENT_CLASS = GPTOutputs
CLIENT_KWARGS = dict(
    temperature=TEMPERATURE,
    model_id="gpt-3.5-turbo-0125",
)

POSSIBLE_DATASETS = [
    "breast_cancer",
    "california_housing",
    "heart_disease",
    "wine_quality",
    "fake_data",
]

for ds in DATASETS:
    if ds not in POSSIBLE_DATASETS:
        raise ValueError(
            f"Dataset {ds} not recognized. Possible datasets are {POSSIBLE_DATASETS}"
        )

DATASET_FUNCTIONS = {
    "fake_data": load_fake_data,
    "breast_cancer": load_breast_cancer,
    "california_housing": load_california_housing,
    "heart_disease": load_heart_disease,
    "wine_quality": load_wine_quality,
}

DATASET_MODEL_TYPES = {
    "fake_data": "linear",
    "breast_cancer": "logistic",
    "california_housing": "linear",
    "heart_disease": "logistic",
    "wine_quality": "logistic",
}


DATASET_SPLIT_CLASSES = {
    "fake_data": skms.ShuffleSplit,
    "breast_cancer": skms.StratifiedShuffleSplit,
    "uti": skms.StratifiedShuffleSplit,
    "california_housing": skms.ShuffleSplit,
    "heart_disease": skms.StratifiedShuffleSplit,
    "wine_quality": skms.StratifiedShuffleSplit,
}


PROMPTS_DIR = "./prompts/icl"

Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)


# functions to process the outputs
def process_posteriors_mle_and_samples(
    outputs: t.List[t.Tuple[np.ndarray, np.ndarray, t.Tuple[np.ndarray, np.ndarray]]]
) -> t.Tuple[np.ndarray, np.ndarray, t.Tuple[np.ndarray, np.ndarray]]:
    """
    Processes the output from multiple calls of
    :code:`sample_approximate_gpt_internal_predictive_model_parameters`
    into a format useful for analysis. This function essentially
    stacks each of the sets of outputs into a single numpy array.


    Arguments
    ----------

    outputs : list of tuples
        The outputs of the function

    Returns
    ---------

    posterior_parameter_samples : np.ndarray
        The prior parameter samples

    mle_loss : np.ndarray
        The maximum likelihood loss

    sample : tuple of np.ndarray
        The samples


    """

    posterior_parameter_samples, mle_loss, x_sample, y_sample = [], [], [], []
    for repeat_output in outputs:
        (
            posterior_parameter_samples_repeat,
            mle_loss_repeat,
            x_sample_repeat,
            y_sample_repeat,
        ) = (
            [],
            [],
            [],
            [],
        )
        for output in repeat_output:
            posterior_parameter_samples_repeat.append(make_list(output[0]))
            mle_loss_repeat.append(make_list(output[1]))
            x_sample_repeat.append(make_list(output[2][0]))
            y_sample_repeat.append(make_list(output[2][1]))

        posterior_parameter_samples.append(posterior_parameter_samples_repeat)
        mle_loss.append(mle_loss_repeat)
        x_sample.append(x_sample_repeat)
        y_sample.append(y_sample_repeat)

    sample = [x_sample, y_sample]

    return posterior_parameter_samples, mle_loss, sample


def save_posteriors_mle_samples_and_prompts(
    posterior_parameter_samples: np.ndarray,
    mle_loss: np.ndarray,
    train_idx: t.List[np.ndarray],
    path: str,
):
    """
    Given the outputs from :code:`process_posteriors_mle_and_samples`,
    this function will save them to disk in the specified path as json files.
    If the path doesn't exist, it will be created.

    Arguments
    ----------

    posterior_parameter_samples : np.ndarray
        The prior parameter samples

    mle_loss : np.ndarray
        The maximum likelihood loss

    train_idx: list of np.ndarray
        The training indices

    path : str
        The path to save the outputs


    Returns
    ---------

    None

    """

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "posterior_parameter_samples.json"), "w") as f:
        json.dump(posterior_parameter_samples, f)

    with open(os.path.join(path, "mle_loss.json"), "w") as f:
        json.dump(mle_loss, f)

    with open(os.path.join(path, "train_idx.json"), "w") as f:
        json.dump(train_idx, f)


# iterating over the dataset to get the internal model
for dataset_name in tqdm.tqdm(
    DATASETS,
    total=len(DATASETS),
    desc="Iterating over datasets",
    position=0,
):

    # reproducibility (on our side! GPT-4 is not always deterministic)
    SEED += 1
    rng = np.random.default_rng(SEED)

    client = CLIENT_CLASS(**CLIENT_KWARGS)

    # getting the dataset and information for the language model prompt
    dataset = DATASET_FUNCTIONS[dataset_name](as_frame=True)
    model_type = DATASET_MODEL_TYPES[dataset_name]
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

    feature_names = dataset.feature_names[:N_FEATURES]

    if N_DATAPOINTS_IN_SAMPLE < len(feature_names):
        raise ValueError(
            f"The number of datapoints in the sample should be greater than "
            f"the number of features in the dataset. Got {N_DATAPOINTS_IN_SAMPLE} "
            f"datapoints and {len(feature_names)} features."
        )

    # getting data
    dataset = DATASET_FUNCTIONS[dataset_name](as_frame=False)
    feature_names = dataset.feature_names[:N_FEATURES]
    X, y = dataset["data"], dataset["target"]
    X, y = X[:, :N_FEATURES], y

    splitter_class = DATASET_SPLIT_CLASSES[dataset_name]

    splits = splitter_class(
        n_splits=N_REPEATS,
        train_size=N_TRAINING_POINTS,
        random_state=rng.integers(1e6),
    ).split(X, y)

    train_idxs = [train_idx for train_idx, _ in splits]

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
    else:
        preprocessing = skpipe.Pipeline(
            [
                (
                    "passthrough",
                    skpre.FunctionTransformer(
                        func=lambda x: x,
                        inverse_func=lambda x: x,
                    ),
                ),
            ]
        )

    # approximating the language model's internal model for
    # each of the prompts

    outputs = []
    for train_idx in tqdm.tqdm(
        train_idxs,
        total=len(train_idxs),
        desc=f"Splitting the dataset {dataset_name}",
        position=1,
    ):
        outputs_data_split = []
        for sr, fm in tqdm.tqdm(
            itertools.product(system_roles, final_messages),
            total=len(system_roles) * len(final_messages),
            desc=f"Getting outputs for dataset {dataset_name}",
            position=2,
        ):
            output_one_experiment = (
                sample_approximate_llm_internal_predictive_model_parameters(
                    client=client,
                    n_samples=N_SAMPLES,
                    n_datapoints_in_sample=N_DATAPOINTS_IN_SAMPLE,
                    required_model=model_type,
                    system_role=sr,
                    final_message=fm,
                    feature_names=feature_names,
                    rng=rng,
                    demonstration=[
                        preprocessing.fit_transform(X[train_idx]),
                        y[train_idx],
                    ],
                    x_sample_low=X_SAMPLE_LOW,
                    x_sample_high=X_SAMPLE_HIGH,
                    return_mle_loss_and_samples=True,
                    dry_run=False,
                    verbose=False,
                )
            )
            outputs_data_split.append(output_one_experiment)

        outputs.append(outputs_data_split)


    # saving the outputs along with the prompts
    save_path = os.path.join(SAVE_PATH, dataset_name)

    posterior_parameter_samples, mle_loss, sample = process_posteriors_mle_and_samples(
        outputs
    )
    save_posteriors_mle_samples_and_prompts(
        posterior_parameter_samples=posterior_parameter_samples,
        mle_loss=mle_loss,
        train_idx=[train_idx.tolist() for train_idx in train_idxs],
        path=save_path,
    )

    print(f"Saved the posteriors and samples for dataset {dataset_name} to {save_path}")
