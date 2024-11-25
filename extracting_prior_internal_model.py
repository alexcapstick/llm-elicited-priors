import os
import json
import typing as t
import tqdm
import argparse
import numpy as np
import itertools
from pathlib import Path

from my_code.utils import make_list
from my_code.datasets import (
    load_fake_data,
    load_breast_cancer,
    load_uti,
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
parser.add_argument(
    "--dataset",
    type=str,
    default=[
        "breast_cancer",
        "uti",
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
    default="./priors/internal_model/prior_with_multiple_messages/",
)
args = parser.parse_args()

#### the number of features to use in the dataset.
#### this is because some of the language models have trouble predicting
#### the correct output when the number of features is too large
N_FEATURES = 10


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
    "uti",
    "california_housing",
    "heart_disease",
    "wine_quality",
    "fake_data",
]

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

PROMPTS_DIR = "./prompts/icl"

Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)


# functions to process the outputs
def process_priors_mle_and_samples(
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

    prior_parameter_samples : np.ndarray
        The prior parameter samples

    mle_loss : np.ndarray
        The maximum likelihood loss

    sample : tuple of np.ndarray
        The samples


    """

    prior_parameter_samples, mle_loss, x_sample, y_sample = [], [], [], []
    for output in outputs:
        prior_parameter_samples.append(make_list(output[0]))
        mle_loss.append(make_list(output[1]))
        x_sample.append(make_list(output[2][0]))
        y_sample.append(make_list(output[2][1]))

    sample = [x_sample, y_sample]

    return prior_parameter_samples, mle_loss, sample


def save_priors_mle_samples_and_prompts(
    prior_parameter_samples: np.ndarray,
    mle_loss: np.ndarray,
    samples: t.Tuple[np.ndarray, np.ndarray],
    path: str,
):
    """
    Given the outputs from :code:`process_priors_mle_and_samples`,
    this function will save them to disk in the specified path as json files.
    If the path doesn't exist, it will be created.

    Arguments
    ----------

    prior_parameter_samples : np.ndarray
        The prior parameter samples

    mle_loss : np.ndarray
        The maximum likelihood loss

    samples : tuple of np.ndarray
        The samples. If :code:`None`, they will not
        be saved.

    path : str
        The path to save the outputs


    Returns
    ---------

    None

    """

    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, "prior_parameter_samples.json"), "w") as f:
        json.dump(prior_parameter_samples, f)

    with open(os.path.join(path, "mle_loss.json"), "w") as f:
        json.dump(mle_loss, f)

    if samples is not None:
        with open(os.path.join(path, "samples.json"), "w") as f:
            json.dump(samples, f)


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

    # approximating the language model's internal model for
    # each of the prompts

    outputs = []

    for sr, fm in tqdm.tqdm(
        itertools.product(system_roles, final_messages),
        total=len(system_roles) * len(final_messages),
        desc=f"Getting outputs for dataset {dataset_name}",
        position=1,
    ):

        outputs.append(
            sample_approximate_llm_internal_predictive_model_parameters(
                client=client,
                n_samples=N_SAMPLES,
                n_datapoints_in_sample=N_DATAPOINTS_IN_SAMPLE,
                required_model=model_type,
                system_role=sr,
                final_message=fm,
                feature_names=feature_names,
                rng=rng,
                demonstration=None,
                x_sample_low=X_SAMPLE_LOW,
                x_sample_high=X_SAMPLE_HIGH,
                return_mle_loss_and_samples=True,
                dry_run=False,
                verbose=False,
            )
        )

        # saving the outputs along with the prompts
        save_path = os.path.join(SAVE_PATH, dataset_name)

        prior_parameter_samples, mle_loss, sample = process_priors_mle_and_samples(
            outputs
        )

        save_priors_mle_samples_and_prompts(
            prior_parameter_samples=prior_parameter_samples,
            mle_loss=mle_loss,
            path=save_path,
            samples=sample if dataset_name == "fake_data" else None,
        )

    print(
        f"Finished saving the priors and samples for dataset {dataset_name} to {save_path}"
    )
