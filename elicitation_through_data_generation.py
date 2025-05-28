import os
from pathlib import Path
import json
import tqdm
import numpy as np
import pymc as pm
from dataclasses import dataclass
from typing import List, Tuple
import sklearn.model_selection as skms
import sklearn.preprocessing as skpre
import sklearn.impute as skimpute
import sklearn.pipeline as skpipe
import sklearn.metrics as skmetrics
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
import argparse
import pickle
import scipy.stats

from openai import OpenAI

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

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="The dataset to use for the experiments",
    nargs="+",
    default=[
        "fake_data",
        "uti",
        "breast_cancer",
        "heart_disease",
        "diabetes",
        "hypothyroid",
        # "california_housing",
        # "wine_quality",
    ],
)

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
    "--model",
    type=str,
    help="The model to use for the experiments",
    default="gpt-3.5-turbo-0125",
)


args = parser.parse_args()

for dataset in args.dataset:
    if dataset not in [
        "fake_data",
        "uti",
        "breast_cancer",
        "california_housing",
        "heart_disease",
        "wine_quality",
        "diabetes",
        "hypothyroid",
    ]:
        raise ValueError(f"Dataset {dataset} not recognised")


POSSIBLE_MODELS = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-2024-04-09",
]

if args.model not in POSSIBLE_MODELS:
    raise ValueError(f"Model {args.model} not recognised")


MODEL = args.model

PRIORS_DIR = Path("./priors/generated_data").joinpath(
    MODEL.replace("/", "-").replace(".", "-").replace(":", "-").lower()
)
PRIORS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = "./results/generated_data_mcmc"
RESULTS_DIR = Path(RESULTS_DIR).joinpath(
    MODEL.replace("/", "-").replace(".", "-").replace(":", "-").lower()
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 2

##### for priors
GET_FROM_API = args.get_priors
DATASET_PRIORS_TO_GET = args.dataset
NUMBER_PRIOR_SAMPLES = 80

##### for experiments
RUN_EXPERIMENTS = args.run_mcmc
N_SPLITS = 10
N_DATA_POINTS_SEEN = [2, 5, 10, 20, 30, 40, 50]
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

print("using the directory for priors:", PRIORS_DIR)
print("using the directory for results:", RESULTS_DIR)


@dataclass
class Attribute:
    name: str
    description: str
    dtype: str
    values: List[str] | None


@dataclass
class MetaData:
    name: str
    description: str
    field: str
    features: List[Attribute]
    target: Attribute
    task_type: str


#### the following is from the baseline code:
# https://github.com/henrygouk/llm-prior/tree/main
class LLMSampler:
    def __init__(self, client: OpenAI, model: str, meta_data):
        self.client = client
        self.model = model
        self.meta_data = meta_data
        self.features_schema = self._create_features_schema()

    def _create_features_schema(self) -> dict:
        properties = {}

        for f in self.meta_data.features:
            if f.dtype == "float":
                properties[f.name] = {"type": "number"}
            elif f.dtype == "str":
                properties[f.name] = {"type": "string", "enum": f.values}
            else:
                raise ValueError(
                    f"Invalid data type: {f.dtype}. Must be one of ['float', 'str']"
                )

        return {"type": "object", "properties": properties}

    def _features_dict_to_list(self, features: dict) -> List:
        return [features.get(f.name, 0.0) for f in self.meta_data.features]

    ## add json scheme to prompt for gpt-3.5 support
    def _sample_features_batch(self, n: int) -> List:
        schema = {"type": "array", "items": self.features_schema}

        if self.meta_data.task_type == "classification":
            target_possible_values = ", ".join(self.meta_data.target.values)
        elif self.meta_data.task_type == "regression":
            target_possible_values = "float values"

        nl = "\n"
        messages = [
            {
                "role": "system",
                "content": f"You are an expert in the field of {self.meta_data.field}.\n"
                f"Your top priority is to provide statisticians with the domain knowedge required to analyse their data. {self.meta_data.description}\n"
                f"The dataset has the following features:\n{nl.join([f.name + ': ' + f.description for f in self.meta_data.features])}.\n"
                f"The dataset has the following target:\n{self.meta_data.target.name}: {self.meta_data.target.description}. "
                f"It can take these values: {target_possible_values}.\n"
                f"You will return the information in the following JSON schema: {schema}",
            },
            {
                "role": "user",
                "content": f"Give {n} rows of example data from a variety of targets in JSON format.\n",
            },
        ]

        print("sending messages:", messages)

        n_retries = 0

        max_n_retries = 10

        while n_retries < max_n_retries:

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1.0,
                    # extra_body={"guided_json": schema},
                    response_format={"type": "json_object"},
                )

                print("model response:", response.choices[0].message.content)
                print("to json:", json.loads(response.choices[0].message.content))

                functions_to_try = [
                    lambda x: eval(x)["examples"],
                    lambda x: eval(x)["example"],
                    lambda x: eval(x)["data"],
                    lambda x: eval(x)["items"],
                    lambda x: eval(x)["item"],
                    lambda x: eval(x)["sampleData"],
                    lambda x: eval(x),
                ]

                response_to_extract_from = response.choices[0].message.content
                print("features:", response_to_extract_from)

                able_to_extract_features = False
                for f in functions_to_try:
                    try:
                        data_points = f(response_to_extract_from)
                        able_to_extract_features = True
                        break
                    except Exception as e:
                        pass

                if not able_to_extract_features:
                    print("failed to extract features")
                    print("retrying")
                    n_retries += 1

                else:
                    output = [self._features_dict_to_list(x) for x in data_points]
                    return output

            except Exception as e:
                print("failed to extract features")
                print("retrying")
                n_retries += 1

            if n_retries == max_n_retries:
                raise ValueError("failed to extract features")

    def sample_features(self, n: int, batch_size=1) -> np.ndarray:
        X = []

        while len(X) < n:
            batch = self._sample_features_batch(batch_size)
            print("\n\n\n")
            print("batch:", batch)
            print("\n\n\n")
            if len(batch) != 0:
                X.extend(batch)

        return np.array(X[:n])

    ##### added: allow for sampling regression targets and retries when failing
    def _sample_target_single(self, x: np.ndarray, num_trials: int = 1) -> List[int]:
        schema = {"type": "string", "enum": self.meta_data.target.values}

        if self.meta_data.task_type == "classification":
            target_possible_values = (
                "'" + "' or '".join(self.meta_data.target.values) + "'"
            )
        elif self.meta_data.task_type == "regression":
            target_possible_values = "float values"

        nl = "\n"
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert in the field of {self.meta_data.field}.\n"
                    f"Your top priority is to provide statisticians with the domain knowedge required to analyse their data. {self.meta_data.description}\n"
                    f"The dataset has the following features:\n{nl.join([f.name + ': ' + f.description for f in self.meta_data.features])}.\n"
                    f"The dataset has the following target:\n{self.meta_data.target.name}: {self.meta_data.target.description}. "
                    + (
                        f"The target can take only one of these values: {target_possible_values}.\n"
                        if self.meta_data.task_type == "classification"
                        else "The target can take float values, please only return the target and no other text.\n"
                    )
                    # + f"Please only provide the target.\n"
                    + f"By replacing {self.meta_data.target.values} with your chosen target, "
                    f"you will return the information in the following JSON schema: {schema}.\n"
                ),
            },
            {
                "role": "user",
                "content": f"Give the target value for the row in the dataset:\n"
                f"{' '.join(['The ' + self.meta_data.features[i].name + ' is ' + str(x[i]) + '.' for i in range(len(x))])}\n"
                + (
                    f"Please only provide one target, not multiple targets. This target can either be {target_possible_values}, not both."
                    f"If you are unsure, please provide your best guess.\n"
                    if self.meta_data.task_type == "classification"
                    else "The target can take float values, please only return the target and no other text.\n"
                ),
            },
        ]

        functions_to_try = [
            lambda x: eval(x)["target"],
            # sometimes the model returns a list of targets with only one element
            lambda x: (
                eval(x)["target"][0]
                if (type(eval(x)["target"]) == list and len(eval(x)["target"]) == 1)
                else eval(x)
            ),
            lambda x: eval(x)["enum"],
            # sometimes the model returns a list of targets with only one element
            lambda x: (
                eval(x)["enum"][0]
                if (type(eval(x)["enum"]) == list and len(eval(x)["enum"]) == 1)
                else eval(x)
            ),
            lambda x: eval(x)["uti diagnosis"],
            lambda x: (
                eval(x)["uti diagnosis"][0]
                if (
                    type(eval(x)["uti diagnosis"]) == list
                    and len(eval(x)["uti diagnosis"]) == 1
                )
                else eval(x)
            ),
            lambda x: eval(x),
        ]

        n_retries = 0
        max_n_retries = 10

        while n_retries < max_n_retries:

            print("sending messages:", messages)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                n=num_trials,
                temperature=1.0,
                # extra_body={"guided_json": schema},
                response_format={"type": "json_object"},
            )

            responses_to_extract_from = response.choices
            print("model target response:", responses_to_extract_from)

            # transforming the targets, which might be in the wrong form
            targets = []
            all_conversions = []
            for choice in responses_to_extract_from:
                conversion_passed = False
                choice = choice.message.content
                print(choice)
                for f in functions_to_try:
                    try:
                        # get the index of the target
                        returned_target = f(choice)

                        if self.meta_data.task_type == "classification":
                            targets.append(
                                self.meta_data.target.values.index(returned_target)
                            )
                        elif self.meta_data.task_type == "regression":
                            targets.append(float(returned_target))
                        conversion_passed = True
                        all_conversions.append(True)
                        break

                    except Exception as e:
                        pass

                # if we didn't find a conversion, we need to retry
                if not conversion_passed:
                    print("failed to convert target")
                    print("choice:", choice)
                    print("retrying")
                    n_retries += 1
                    break

            # if we converted all targets, we can return
            if np.sum(all_conversions) == num_trials:
                print("converted all targets")
                print("targets:", targets)
                return targets

        if n_retries == max_n_retries:
            raise ValueError("failed to convert target")

    ##### added: allow for sampling regression targets
    def sample_targets(
        self, X: np.ndarray, num_trials: int = 1, target_smooth: float = 0.5
    ) -> np.ndarray:

        if self.meta_data.task_type == "classification":
            y = np.ones((X.shape[0], len(self.meta_data.target.values))) * target_smooth

            for i in range(X.shape[0]):
                for j in self._sample_target_single(X[i], num_trials):
                    y[i, j] += 1

            return y / y.sum(axis=1, keepdims=True)

        elif self.meta_data.task_type == "regression":
            y = np.zeros((X.shape[0], 1))
            for i in range(X.shape[0]):
                y[i] = np.mean(self._sample_target_single(X[i], num_trials))
            return y

    ### split up so can save arrays separately
    def sample_X(
        self,
        n: int,
        batch_size: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = self.sample_features(n, batch_size)
        return X

    def sample_y(
        self,
        X: np.ndarray,
        num_trials: int = 10,
        target_smooth: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y = self.sample_targets(X, num_trials, target_smooth)
        return y


#### the following is from the baseline code:
# https://github.com/henrygouk/llm-prior/tree/main
class BayesLogisticRegression:

    def __init__(
        self,
        tau: float = 1.0,
        gamma: float = 0.5,
        delta: float = 1.0,
        rng=None,
        n_samples=1000,
        n_chains=4,
    ):
        self.tau = tau
        self.gamma = gamma
        self.delta = delta
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        self.n_samples = n_samples
        self.n_chains = n_chains

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        K_X: np.ndarray | None = None,
        K_y: np.ndarray | None = None,
        progressbar: bool = True,
        num_classes: int = None,
    ):
        self.model = pm.Model()

        with self.model:
            self.num_trials = X.shape[0]
            self.num_features = X.shape[1]

            if num_classes is None:
                self.num_classes = len(np.unique(y))
            else:
                self.num_classes = num_classes

            coords = {
                "trials": np.arange(self.num_trials),
                "features": np.arange(self.num_features),
                "classes": np.arange(self.num_classes),
            }

            self.model = pm.Model(coords=coords)

            with self.model:
                X_sym = pm.Data("X", X, dims=("trials", "features"))
                y_sym = pm.Data("y", y, dims=("trials"))

                beta = pm.Normal(
                    "beta",
                    mu=0,
                    tau=self.tau,
                    shape=(self.num_features, self.num_classes),
                )
                alpha = pm.Normal("alpha", mu=0, tau=self.tau, shape=self.num_classes)

                logits = pm.math.dot(X_sym, beta) + alpha
                y_obs = pm.Categorical(
                    "y_obs", p=pm.math.softmax(logits, axis=1), observed=y_sym
                )

                if K_X is not None and K_y is not None:
                    K_logits = pm.math.dot(K_X, beta) + alpha
                    a = self.gamma + self.delta * pm.math.softmax(K_logits, axis=1)
                    K_y_obs = pm.Dirichlet("K_y_obs", a=a, observed=K_y)

                self.trace = pm.sample(
                    self.n_samples,
                    tune=1000,
                    chains=self.n_chains,
                    progressbar=progressbar,
                    random_seed=self.rng,
                )

    def predict(self, X: np.ndarray, progressbar: bool = False) -> np.ndarray:
        # Sample from the posterior predictive
        with self.model:
            pm.set_data(
                {"X": X, "y": np.zeros(X.shape[0], dtype=int)},
                coords={
                    "trials": np.arange(self.num_trials, self.num_trials + X.shape[0]),
                    "features": np.arange(X.shape[1]),
                },
            )

            post_pred = pm.sample_posterior_predictive(
                self.trace,
                predictions=True,
                progressbar=progressbar,
                random_seed=rng,
            ).predictions["y_obs"]
            return post_pred.to_numpy()

            # votes = post_pred.to_numpy().reshape((-1, X.shape[0])).T
            # return scipy.stats.mode(votes, axis=1).mode


###################### edit this class as it is currently logistic regression
#### the following is from the baseline code:
# https://github.com/henrygouk/llm-prior/tree/main
class BayesLinearRegression:

    def __init__(
        self,
        tau: float = 1.0,
        delta: float = 1.0,
        rng=None,
        n_samples=1000,
        n_chains=4,
    ):
        self.tau = tau
        self.delta = delta
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
        self.n_samples = n_samples
        self.n_chains = n_chains

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        K_X: np.ndarray | None = None,
        K_y: np.ndarray | None = None,
        progressbar: bool = True,
        num_classes: int = None,
    ):
        self.model = pm.Model()

        if len(y.shape) > 1:
            y = y.ravel()

        if len(K_y.shape) > 1:
            K_y = K_y.ravel()

        with self.model:
            self.num_trials = X.shape[0]
            self.num_features = X.shape[1]

            coords = {
                "trials": np.arange(self.num_trials),
                "features": np.arange(self.num_features),
            }

            self.model = pm.Model(coords=coords)

            with self.model:
                X_sym = pm.Data("X", X, dims=("trials", "features"))
                y_sym = pm.Data("y", y, dims=("trials"))

                likelihood_sigma = pm.HalfCauchy(
                    "sigma",
                    beta=1,
                )
                beta = pm.Normal(
                    "beta",
                    mu=0,
                    tau=self.tau,
                    shape=(self.num_features,),
                )
                alpha = pm.Normal("alpha", mu=0, tau=self.tau, shape=(1,))

                outcomes = pm.Normal(
                    "y_obs",
                    mu=pm.math.dot(X_sym, beta) + alpha,
                    sigma=likelihood_sigma,
                    observed=y_sym,
                    dims=("trials",),
                )

                if K_X is not None and K_y is not None:
                    K_outcomes = pm.Normal(
                        "K_y_obs",
                        mu=self.delta * (pm.math.dot(K_X, beta) + alpha),
                        sigma=likelihood_sigma,
                        observed=K_y,
                    )

                self.trace = pm.sample(
                    self.n_samples,
                    tune=1000,
                    chains=self.n_chains,
                    progressbar=progressbar,
                    random_seed=rng,
                )

    def predict(self, X: np.ndarray, progressbar: bool = False) -> np.ndarray:
        # Sample from the posterior predictive
        with self.model:
            pm.set_data(
                {"X": X, "y": np.zeros(X.shape[0], dtype=int)},
                coords={
                    "trials": np.arange(self.num_trials, self.num_trials + X.shape[0]),
                    "features": np.arange(X.shape[1]),
                },
            )

            post_pred = (
                pm.sample_posterior_predictive(
                    self.trace,
                    predictions=True,
                    progressbar=progressbar,
                    random_seed=rng,
                )
                .predictions["y_obs"]
                .to_numpy()
            )
            return post_pred


uti_meta_data = MetaData(
    name="uti",
    description="The dataset contains measurements obtained from sensors around a home, and the goal is to use these measurements to determine a urinary tract infection diagnosis.",
    field="healthcare",
    task_type="classification",
    features=[
        Attribute(
            "bathroom frequency",
            "The number of trips to the bathroom in a day.",
            "float",
            None,
        ),
        Attribute(
            "bedroom frequency",
            "The number of trips to the bedroom in a day.",
            "float",
            None,
        ),
        Attribute(
            "night time awake frequency",
            "The number of times the patient wakes up at night.",
            "float",
            None,
        ),
        Attribute(
            "mean night time heart rate",
            "The mean heart rate during the night.",
            "float",
            None,
        ),
        Attribute(
            "standard deviation of night time heart rate",
            "The standard deviation of the heart rate during the night.",
            "float",
            None,
        ),
        Attribute(
            "mean night time respiratory rate",
            "The mean respiratory rate during the night.",
            "float",
            None,
        ),
        Attribute(
            "standard deviation of night time respiratory rate",
            "The standard deviation of the respiratory rate during the night.",
            "float",
            None,
        ),
        Attribute(
            "night time bathroom frequency",
            "The number of trips to the bathroom at night.",
            "float",
            None,
        ),
        Attribute(
            "daytime bathroom frequency",
            "The number of trips to the bathroom during the day.",
            "float",
            None,
        ),
        Attribute(
            "number of previous urinary tract infections",
            "The number of previous urinary tract infections.",
            "float",
            None,
        ),
        Attribute(
            "sex (male = 0, female = 1)", "the sex of the participant.", "float", None
        ),
    ],
    target=Attribute(
        "uti diagnosis",
        "The diagnosis of a urinary tract infection.",
        "str",
        ["negative", "positive"],
    ),
)

synthetic_task_meta_data = MetaData(
    name="fake_data",
    description="The dataset contains features and targets related through the equation 'target' = 2 * 'feature 0' - 1 * 'feature 1' + 1 * 'feature 2'.",
    field="synthetic data generation",
    task_type="regression",
    features=[
        Attribute(
            "feature 0",
            "The first feature.",
            "float",
            None,
        ),
        Attribute(
            "feature 1",
            "The second feature.",
            "float",
            None,
        ),
        Attribute(
            "feature 2",
            "The third feature.",
            "float",
            None,
        ),
    ],
    target=Attribute(
        "target",
        "The target variable.",
        "float",
        None,
    ),
)

heart_disease_meta_data = MetaData(
    name="heart_disease",
    description="The dataset contains physiological characteristics related to heart disease diagnosis.",
    field="healthcare",
    task_type="classification",
    features=[
        Attribute(
            "age",
            "The age of the patient.",
            "float",
            None,
        ),
        Attribute(
            "sex (1 = male, 0 = female)", "the sex of the participant.", "float", None
        ),
        Attribute(
            "resting blood pressure (on admission to the hospital)",
            "The resting blood pressure of the patient.",
            "float",
            None,
        ),
        Attribute(
            "serum cholestoral in mg/dl",
            "The serum cholestoral of the patient.",
            "float",
            None,
        ),
        Attribute(
            "fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
            "The fasting blood sugar of the patient.",
            "float",
            None,
        ),
        Attribute(
            "resting electrocardiographic results (1 = abnormal, 0 = normal)",
            "The resting electrocardiographic results of the patient.",
            "float",
            None,
        ),
        Attribute(
            "maximum heart rate achieved",
            "The maximum heart rate achieved by the patient.",
            "float",
            None,
        ),
        Attribute(
            "exercise induced angina",
            "Whether the patient has exercise induced angina.",
            "float",
            None,
        ),
        Attribute(
            "ST depression induced by exercise relative to rest",
            "The ST depression induced by exercise relative to rest.",
            "float",
            None,
        ),
        Attribute(
            "number of major vessels (0 - 3) colored by flourosopy",
            "The number of major vessels colored by flourosopy.",
            "float",
            None,
        ),
    ],
    target=Attribute(
        "heart disease diagnosis",
        "The diagnosis of heart disease.",
        "str",
        ["negative", "positive"],
    ),
)

hypothyroid_meta_data = MetaData(
    name="hypothyroid",
    description="The dataset contains blood test results related to hypothyroid diagnosis.",
    field="healthcare",
    task_type="classification",
    features=[
        Attribute(
            "thyroid-stimulating hormone value (TSH)",
            "The TSH value of the patient.",
            "float",
            None,
        ),
        Attribute(
            "triiodothyronine value (T3)", "The T3 value of the patient.", "float", None
        ),
        Attribute(
            "total thyroxine value (TT4)",
            "The TT4 value of the patient.",
            "float",
            None,
        ),
        Attribute(
            "thyroxine uptake value (T4U)",
            "The T4U value of the patient.",
            "float",
            None,
        ),
    ],
    target=Attribute(
        "hypothyroid diagnosis",
        "The diagnosis of hypothyroid.",
        "str",
        ["normal", "hypothyroidism"],
    ),
)

diabetes_meta_data = MetaData(
    name="diabetes",
    description="The dataset relates the quantitative progression of diabetes one year after the initial assessment to physiological traits recorded at the baseline assessment.",
    field="healthcare",
    task_type="regression",
    features=[
        Attribute(
            "age",
            "The age of the patient.",
            "float",
            None,
        ),
        Attribute(
            "sex",
            "The sex of the patient.",
            "float",
            None,
        ),
        Attribute(
            "body mass index",
            "The body mass index of the patient.",
            "float",
            None,
        ),
        Attribute(
            "average blood pressure",
            "The average blood pressure of the patient.",
            "float",
            None,
        ),
        Attribute(
            "total serum cholesterol",
            "The total serum cholesterol of the patient.",
            "float",
            None,
        ),
        Attribute(
            "low-density lipoproteins",
            "The low-density lipoproteins of the patient.",
            "float",
            None,
        ),
        Attribute(
            "high-density lipoproteins",
            "The high-density lipoproteins of the patient.",
            "float",
            None,
        ),
        Attribute(
            "total cholesterol / HDL",
            "The total cholesterol / HDL of the patient.",
            "float",
            None,
        ),
        Attribute(
            "log of serum triglycerides level",
            "The log of the serum triglycerides level of the patient.",
            "float",
            None,
        ),
        Attribute(
            "blood sugar level",
            "The blood sugar level of the patient.",
            "float",
            None,
        ),
    ],
    target=Attribute(
        "progression of diabetes",
        "A quantitative measure of diabetes disease progression one year after baseline (0 to 10)",
        "float",
        None,
    ),
)

breast_cancer_meta_data = MetaData(
    name="breast cancer",
    description="The dataset contains tumour characteristics and their classification as benign or malignant.",
    field="healthcare",
    task_type="classification",
    features=[
        Attribute(
            "mean radius",
            "The mean radius of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "mean texture",
            "The mean texture of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "mean perimeter",
            "The mean perimeter of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "mean area",
            "The mean area of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "mean smoothness",
            "The mean smoothness of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "mean compactness",
            "The mean compactness of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "mean concavity",
            "The mean concavity of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "mean concave points",
            "The mean concave points of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "mean symmetry",
            "The mean symmetry of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "mean fractal dimension",
            "The mean fractal dimension of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "radius error",
            "The radius error of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "texture error",
            "The texture error of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "perimeter error",
            "The perimeter error of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "area error",
            "The area error of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "smoothness error",
            "The smoothness error of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "compactness error",
            "The compactness error of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "concavity error",
            "The concavity error of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "concave points error",
            "The concave points error of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "symmetry error",
            "The symmetry error of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "fractal dimension error",
            "The fractal dimension error of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "worst radius",
            "The worst radius of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "worst texture",
            "The worst texture of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "worst perimeter",
            "The worst perimeter of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "worst area",
            "The worst area of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "worst smoothness",
            "The worst smoothness of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "worst compactness",
            "The worst compactness of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "worst concavity",
            "The worst concavity of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "worst concave points",
            "The worst concave points of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "worst symmetry",
            "The worst symmetry of the tumour.",
            "float",
            None,
        ),
        Attribute(
            "worst fractal dimension",
            "The worst fractal dimension of the tumour.",
            "float",
            None,
        ),
    ],
    target=Attribute(
        "breast cancer diagnosis",
        "The diagnosis of breast cancer.",
        "str",
        ["benign", "malignant"],
    ),
)


meta_data = {
    "uti": uti_meta_data,
    "fake_data": synthetic_task_meta_data,
    "heart_disease": heart_disease_meta_data,
    "hypothyroid": hypothyroid_meta_data,
    "diabetes": diabetes_meta_data,
    "breast_cancer": breast_cancer_meta_data,
}

# get generated data and save it
if GET_FROM_API:
    for dataset_name in DATASET_PRIORS_TO_GET:
        if dataset_name not in meta_data:
            raise ValueError(f"Meta data for {dataset_name} not found")

        print(f"Getting priors for {dataset_name}")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        sampler = LLMSampler(client, MODEL, meta_data[dataset_name])
        K_X = sampler.sample_X(NUMBER_PRIOR_SAMPLES)

        # uncomment (and comment the above line) for debugging the label generation
        # K_X = np.load(
        #     PRIORS_DIR.joinpath(f"{dataset_name}_generated_data_X.npy"),
        # )

        print(f"Saving X priors for {dataset_name}")
        np.save(
            PRIORS_DIR.joinpath(f"{dataset_name}_generated_data_X.npy"),
            K_X,
        )

        K_y = sampler.sample_y(K_X)

        print(f"Saving y priors for {dataset_name}")
        np.save(
            PRIORS_DIR.joinpath(f"{dataset_name}_generated_data_y.npy"),
            K_y,
        )


dataset_models = {
    "fake_data": BayesLinearRegression,
    "uti": BayesLogisticRegression,
    "heart_disease": BayesLogisticRegression,
    "hypothyroid": BayesLogisticRegression,
    "diabetes": BayesLinearRegression,
    "breast_cancer": BayesLogisticRegression,
}

# run experiments with the generated data
if RUN_EXPERIMENTS:

    pbar = tqdm.tqdm(
        total=len(DATASETS_TO_EXPERIMENT) * N_SPLITS * len(N_DATA_POINTS_SEEN),
        desc="Running experiments",
    )

    for dataset_name in DATASETS_TO_EXPERIMENT:

        dataset = DATASET_FUNCTIONS[dataset_name](as_frame=True)

        K_X = np.load(
            PRIORS_DIR.joinpath(f"{dataset_name}_generated_data_X.npy"),
        )
        K_y = np.load(
            PRIORS_DIR.joinpath(f"{dataset_name}_generated_data_y.npy"),
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
            K_X = preprocessing.fit_transform(K_X)
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
            K_X = preprocessing.fit_transform(K_X)
        elif dataset_name == "fake_data":
            pass
        else:
            raise ValueError(f"Preprocessing pipeline for {dataset_name} not defined")

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

        i = 1

        for ns, (train_idx, test_idx) in enumerate(splits):

            results_path = os.path.join(
                RESULTS_DIR,
                f"prior_no_prior_results_{dataset_name}_{ns}.json",
            )

            print(f"Running experiments for {dataset_name}, {ns}")
            print("Results will be saved to:", results_path)

            if Path(results_path).exists():
                with open(results_path, "r") as f:
                    results = json.load(f)
                    print(f"Loaded results for {dataset_name}, {ns}")
            else:
                results = {}

            if dataset_name not in results:
                results[dataset_name] = {}

            if len(results[dataset_name]) > 0:
                results[dataset_name] = {
                    int(k): v for k, v in results[dataset_name].items()
                }

            if ns not in results[dataset_name]:
                results[dataset_name][ns] = {}

            X_train = dataset.data.iloc[train_idx].to_numpy()
            y_train = dataset.target.iloc[train_idx].to_numpy()
            X_test = dataset.data.iloc[test_idx].to_numpy()
            y_test = dataset.target.iloc[test_idx].to_numpy()

            for nps in N_DATA_POINTS_SEEN:

                rng = np.random.default_rng(RANDOM_SEED * i)

                # make ns int instead of str
                if len(results[dataset_name][ns]) > 0:
                    results[dataset_name][ns] = {
                        int(k): v for k, v in results[dataset_name][ns].items()
                    }

                # results[dataset_name][ns][npr][nps] = {}
                if nps not in results[dataset_name][ns]:
                    results[dataset_name][ns][nps] = {}
                else:
                    pbar.update(1)
                    i += 1
                    print(f"Skipping {dataset_name}, {ns}, {nps}")
                    continue

                X_train_seen, y_train_seen = resample(
                    X_train,
                    y_train,
                    n_samples=nps,
                    random_state=rng.integers(1e6),
                    replace=False,
                    stratify=y_train,
                )

                # otherwise scale everything
                if dataset_name != "fake_data":
                    X_train_seen_transformed = preprocessing.fit_transform(X_train_seen)
                    X_test_transformed = preprocessing.transform(X_test)
                else:
                    X_train_seen_transformed = X_train_seen
                    X_test_transformed = X_test

                model = dataset_models[dataset_name](
                    rng=rng,
                    n_samples=N_SAMPLES,
                    n_chains=N_CHAINS,
                )
                model.fit(X_train_seen_transformed, y_train_seen, K_X, K_y)
                y_pred_test = model.predict(X_test_transformed)

                results[dataset_name][ns][nps] = dataset_metrics_function[dataset_name](
                    y_test, y_pred_test
                )
                metric_to_print = dataset_metric_to_print[dataset_name]

                print(
                    "\n",
                    f"Dataset: {dataset_name}, Split: {ns+1}, N Points Seen: {nps},",
                    "\n",
                    f"{results[dataset_name][ns][nps][metric_to_print]},",
                    "\n",
                    "-" * 80,
                    "\n",
                )
                i += 1

                pbar.update(1)

            print("Saving results to:", results_path)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

    pbar.close()
