from pathlib import Path
import pandas as pd
import numpy as np
import sklearn.datasets as skd
from sklearn.utils import Bunch
from ucimlrepo import fetch_ucirepo
import typing


uci_ids = {
    "adult": 2,
    "heart_disease": 45,
    "iris": 53,
    "wine_quality": 186,
    "wine": 109,
}


def load_and_save_uci_dataset(
    dataset_name: str,
    path: Path = Path.home().joinpath("data"),
) -> pd.DataFrame:
    """
    This function loads and saves
    a dataset from the UCI repository
    if it has not been saved before.


    Arguments
    ---------

    dataset_name: str
        The name of the dataset to load.

    path: str or Path
        The directory where the file should be saved.
        Defaults to :code:`Path.home().joinpath("data")`.


    Returns
    -------

    pd.DataFrame
        The dataset. The target column is named :code:`target`.


    """
    file_path = path.joinpath(f"{dataset_name}.parquet")

    if file_path.exists():
        data = pd.read_parquet(file_path)
    else:
        dataset = fetch_ucirepo(id=uci_ids[dataset_name])
        # saving the data for
        # the next time it is loaded
        print("Saving the data")
        data = dataset.data.features.assign(target=dataset.data.targets)
        data.to_parquet(file_path)

    # saving the original dataset as well used in the
    # load_and_save_uci_original_dataset function
    original_file_path = path.joinpath(f"{dataset_name}_original.parquet")
    if not original_file_path.exists():
        original_data = dataset.data.original
        original_data.to_parquet(original_file_path)

    return data


def load_and_save_uci_original_dataset(
    dataset_name: str,
    path: Path = Path.home().joinpath("data"),
) -> pd.DataFrame:
    """
    This function loads and saves
    a dataset from the UCI repository
    if it has not been saved before.

    In particular, it saves the original dataset,
    which may include a different target column name.


    Arguments
    ---------

    dataset_name: str
        The name of the dataset to load.

    path: str or Path
        The directory where the file should be saved.
        Defaults to :code:`Path.home().joinpath("data")`.


    Returns
    -------

    pd.DataFrame
        The dataset, with the original target column name.


    """
    file_path = path.joinpath(f"{dataset_name}_original.parquet")

    if file_path.exists():
        data = pd.read_parquet(file_path)
    else:
        dataset = fetch_ucirepo(id=uci_ids[dataset_name])
        # saving the data for
        # the next time it is loaded
        print("Saving the data")
        data = dataset.data.original
        data.to_parquet(file_path)

    # saving the standard dataset as well used in the
    # load_and_save_uci_dataset function
    standard_file_path = path.joinpath(f"{dataset_name}.parquet")
    if not standard_file_path.exists():
        standard_data = dataset.data.features.assign(target=dataset.data.targets)
        standard_data.to_parquet(standard_file_path)

    return data


def process_original_uti(uti_dataset_original_df: pd.DataFrame) -> pd.DataFrame:

    ## run the following code to process the original dataset:

    # import pandas as pd
    # from my_code.datasets import process_original_uti

    # process_original_uti(
    #     pd.read_parquet(
    #         "~/data/uti_dataset_original.parquet"
    #     )
    # ).to_parquet(
    #     "~/data/uti_dataset.parquet"
    # )

    col_nice_names = {
        "awake_freq": "Night time Awake Frequency",
        "bathroom_daytime_freq": "Daytime Bathroom Frequency",
        "bathroom_daytime_freq_ma": "Daytime Bathroom Frequency Moving Average",
        "bathroom_daytime_freq_ma_delta": "Change in Daytime Bathroom Frequency Moving Average",
        "bathroom_freq": "Bathroom Frequency",
        "bathroom_nighttime_freq": "Night time Bathroom Frequency",
        "bathroom_nighttime_freq_ma": "Night time Bathroom Frequency Moving Average",
        "bathroom_nighttime_freq_ma_delta": "Change in Night time Bathroom Frequency Moving Average",
        "bathroom_relative_transition_time_delta_mean": "Change in Mean Relative Bathroom Transition Time",
        "bathroom_relative_transition_time_delta_std": "Change in Standard Deviation of Relative Bathroom Transition Time",
        "bedroom_freq": "Bedroom Frequency",
        "daily_entropy": "Daily Entropy",
        "hallway_freq": "Hallway Frequency",
        "heart_rate_mean": "Mean Night Time Heart Rate",
        "heart_rate_std": "Standard Deviation of Night Time Heart Rate",
        "kitchen_freq": "Kitchen Frequency",
        "lounge_freq": "Lounge Frequency",
        "previous_uti": "Number of Previous Urinary Tract Infections",
        "respiratory_rate_mean": "Mean Night Time Respiratory Rate",
        "respiratory_rate_std": "Standard Deviation of Night Time Respiratory Rate",
        "sex (male=1, female=0)": "Sex of participant with Male = 1 and Female = 0",
    }

    return (
        uti_dataset_original_df.loc[
            lambda df: df["date"].between(
                pd.to_datetime("2021-07-01"),
                pd.to_datetime("2024-01-01"),
            )
        ]
        .loc[lambda df: ~df["uti_label"].isna()]
        .pipe(
            lambda df: df.replace(
                {
                    "patient_id": {
                        k: str(v) for v, k in enumerate(df["patient_id"].unique())
                    }
                },
            )
        )
        .astype(
            {
                "patient_id": "int",
                "uti_label": "int",
            }
        )
        .reset_index(drop=True)
        .rename(columns=col_nice_names)
        .rename(
            columns={
                "patient_id": "pid",
                "date": "date",
                "uti_label": "target",
            }
        )
        .pipe(lambda df: df.rename(columns={c: c.lower() for c in df.columns}))
    )


def load_uti(
    path: Path = Path.home().joinpath("data"),
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch:
    """
    Load and return the urinary tract infection dataset (uti).

    Arguments
    ---------
    path: str or Path
        The path to the dataset file.
        Defaults to :code:`Path.home().joinpath("data")`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'pid', the patient id for each sample, 'date', the date for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    path = path.joinpath("uti_dataset.parquet")

    uti_dataset_df = (
        pd.read_parquet(path)
        .drop(
            columns=[
                "hallway frequency",
                "kitchen frequency",
                "lounge frequency",
                "night time bathroom frequency moving average",
                "change in night time bathroom frequency moving average",
                "daytime bathroom frequency moving average",
                "change in daytime bathroom frequency moving average",
                "daily entropy",
                "change in mean relative bathroom transition time",
                "change in standard deviation of relative bathroom transition time",
            ]
        )
        .dropna()
    )

    data, target, pid, date, target_names, feature_names = (
        uti_dataset_df.drop(columns=["pid", "date", "target"]),
        uti_dataset_df["target"],
        uti_dataset_df["pid"],
        uti_dataset_df["date"],
        ["no urinary tract infection", "urinary tract infection"],
        uti_dataset_df.drop(columns=["pid", "date", "target"]).columns.tolist(),
    )

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()
        pid = pid.to_numpy()
        date = date.to_numpy()

    if return_X_y:
        return data, target

    uti_dataset = Bunch(
        data=data,
        target=target,
        pid=pid,
        date=date,
        feature_names=np.array(feature_names),
        frame=uti_dataset_df,
        target_names=np.array(target_names),
    )

    return uti_dataset


def load_breast_cancer(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch:
    """
    Load and return the breast cancer dataset.
    Note that the labels of this dataset are flipped compared to
    the sklearn dataset so that malignant=1 and benign=0. This
    is because it makes more sense for calculating recall and precision.

    Arguments
    ---------

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    breast_cancer_dataset = skd.load_breast_cancer(as_frame=True)

    data, target, target_names, feature_names = (
        breast_cancer_dataset["data"],
        # flipping labels so malignant=1 and benign=0 as this makes more sense for
        # calculating recall and precision
        1 - breast_cancer_dataset["target"].astype(int),
        ["benign", "malignant"],
        breast_cancer_dataset.feature_names,
    )

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()

    if return_X_y:
        return data, target

    breast_cancer_dataset = Bunch(
        data=data,
        target=target,
        feature_names=np.array(feature_names),
        frame=breast_cancer_dataset["data"].assign(target=target),
        target_names=np.array(target_names),
    )

    return breast_cancer_dataset


def load_california_housing(
    path: Path = Path.home().joinpath("data"),
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch:
    """
    Load and return the california_housing dataset.

    Arguments
    ---------

    path: str or Path
        Path to the file where the data is stored or should be
        downloaded.
        Defaults to :code:`Path.home().joinpath("data")`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    if not path.joinpath("cal_housing_py3.pkz").exists():
        print("Saving the data")

    california_housing_dataset = skd.fetch_california_housing(
        data_home=path, as_frame=True
    )

    feature_name_map = {
        "MedInc": "median income in block",
        "HouseAge": "median house age in block",
        "AveRooms": "average number of rooms in block",
        "AveBedrms": "average number of bedrooms in block",
        "Population": "block population",
        "AveOccup": "average house occupancy in block",
        "Latitude": "house block latitude",
        "Longitude": "house block longitude",
    }

    data, target, feature_names, target_names, frame = (
        california_housing_dataset["data"],
        california_housing_dataset["target"],
        california_housing_dataset["feature_names"],
        "medium house value (in $100k)",
        (
            california_housing_dataset["frame"]
            .rename(columns=feature_name_map)
            .rename(columns={"MedHouseVal": "medium house value (in $100k)"})
        ),
    )

    feature_names = [feature_name_map[fn] for fn in feature_names]

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()

    if return_X_y:
        return data, target

    california_housing_dataset = Bunch(
        data=data,
        target=target,
        feature_names=np.array(feature_names),
        target_names=np.array([target_names]),
        frame=frame,
    )

    return california_housing_dataset


def load_heart_disease(
    path: Path = Path.home().joinpath("data"),
    return_X_y: bool = False,
    as_frame: bool = False,
):
    """
    Load and return the heart disease dataset: DOI: 10.24432/C52P4X

    Arguments
    ---------

    path: str or Path
        Path to the file where the data is stored or should be
        downloaded.
        Defaults to :code:`Path.home().joinpath("data")`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    features_to_include = [
        "age",
        "sex",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "ca",
    ]

    data = load_and_save_uci_dataset(dataset_name="heart_disease", path=path)
    y = (data["target"] > 0).astype(int).squeeze()
    data = data.drop(columns=["target"])

    # if path.exists():
    #     data = pd.read_parquet(path)
    #     # making sure targets are binary
    #     y = (data["target"] > 0).astype(int).squeeze()
    #     data = data.drop(columns=["target"])
    # else:
    #     heart_disease = fetch_ucirepo(id=45)
    #     # saving the data for
    #     # the next time it is loaded
    #     print("Saving the data")
    #     df = heart_disease.data.features.assign(target=heart_disease.data.targets)
    #     df.to_parquet(path)

    #     data = heart_disease.data.features
    #     # making sure targets are binary
    #     y = (heart_disease.data.target > 0).astype(int).squeeze()

    # selecting only the features we want
    data = data[features_to_include]

    # processing restecg - 0 = normal, 1 = abnormal
    data = data.assign(restecg=(data["restecg"] > 0).astype(int))

    # renaming features with descriptive names

    features_to_rename = {
        "age": "age",
        "sex": "sex (1 = male, 0 = female)",
        "trestbps": "resting blood pressure (on admission to the hospital)",
        "chol": "serum cholestoral in mg/dl",
        "fbs": "fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        "restecg": "resting electrocardiographic results (1 = abnormal, 0 = normal)",
        "thalach": "maximum heart rate achieved",
        "exang": "exercise induced angina",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "ca": "number of major vessels (0 - 3) colored by flourosopy",
    }

    ## renaming the columns
    data = data.rename(columns=features_to_rename)

    # remove rows with missing values in targets or features
    missing = y.isnull().values | data.isnull().any(axis=1).values
    data = data[~missing]
    y = y[~missing]

    # collecting information to return
    data, target, frame, target_names, feature_names, categorical_features = (
        data,
        y,
        data.assign(target=y),
        np.array(["no heart disease", "heart disease"]),
        data.columns.to_numpy(),
        [1, 4, 5, 7, 9],
    )

    if not as_frame:
        data = data.to_numpy()
        target = y.to_numpy()

    if return_X_y:
        return data, target

    heart_disease_dataset = Bunch(
        data=data,
        target=target,
        feature_names=feature_names,
        frame=frame,
        target_names=target_names,
        categorical_features=categorical_features,
    )

    return heart_disease_dataset


def load_wine_quality(
    path: Path = Path.home().joinpath("data"),
    return_X_y: bool = False,
    as_frame: bool = False,
):
    """
    Load and return the wine quality dataset: DOI: 10.24432/C56S3T

    Arguments
    ---------

    path: str or Path
        Path to the file where the data is stored or should be
        downloaded.
        Defaults to :code:`Path.home().joinpath("data")`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    data = load_and_save_uci_dataset(dataset_name="wine_quality", path=path)
    # making sure targets are binary
    y = (data["target"] > 5).astype(int).squeeze()
    data = data.drop(columns=["target"])

    features_to_rename = {
        "fixed_acidity": "fixed acidity",
        "volatile_acidity": "volatile acidity",
        "citric_acid": "citric acid",
        "residual_sugar": "residual sugar",
        "chlorides": "chlorides",
        "free_sulfur_dioxide": "free sulfur dioxide",
        "total_sulfur_dioxide": "total sulfur dioxide",
        "density": "density",
        "pH": "pH",
        "sulphates": "sulphates",
        "alcohol": "alcohol",
    }

    data = data.rename(columns=features_to_rename)

    # remove rows with missing values in targets or features
    missing = y.isnull().values | data.isnull().any(axis=1).values
    data = data[~missing]
    y = y[~missing]

    # collecting information to return
    data, target, frame, target_names, feature_names = (
        data,
        y,
        data.assign(target=y),
        np.array(["bad quality", "good quality"]),
        data.columns.to_numpy(),
    )

    if not as_frame:
        data = data.to_numpy()
        target = y.to_numpy()

    if return_X_y:
        return data, target

    wine_quality_dataset = Bunch(
        data=data,
        target=target,
        feature_names=feature_names,
        frame=frame,
        target_names=target_names,
    )

    return wine_quality_dataset


def load_french_motor_insurance_claims(
    path: Path = Path.home().joinpath("data"),
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch:
    """
    --------
    Unused in the experiments as many of the features are categorical
    and need to be one-hot encoded, which often provides too many features
    for in-context learning to work well with in our testing.
    --------

    Load and return the french_motor_insurance_claims dataset.

    Arguments
    ---------

    path: str or Path
        Path to the file where the data is stored or should be
        downloaded.
        Defaults to :code:`Path.home().joinpath("data")`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    if not path.joinpath("cal_housing_py3.pkz").exists():
        print("Saving the data")

    french_motor_insurance_claims_dataset = skd.fetch_openml(
        data_id=41214, data_home=path, as_frame=True
    )

    feature_name_map = {
        "VehPower": "the power of the car from 0 - 1 (low to high)",
        "VehAge": "the vehicle age (in years)",
        "DrivAge": "the driver age (in years)",
        "BonusMalus": "bonus or malus (True=bonus, False=malus)",
        "VehGas": "car fuel type (0 = regular, 1 = diesel)",
        "Density": "log10 of number of inhabitants per km2 in the city the driver lives in",
    }

    target_names = "number of claims in a year"

    data = french_motor_insurance_claims_dataset.frame[
        ["DrivAge", "BonusMalus", "VehAge", "VehPower", "VehGas", "Density"]
    ]
    target = french_motor_insurance_claims_dataset.frame.apply(
        lambda df: df["ClaimNb"] / df["Exposure"], axis=1
    ).rename(target_names)

    # processing the data
    data = (
        data.astype(object)
        .assign(
            Density=lambda df: np.log10(df["Density"].astype(float)),
            VehGas=lambda df: df["VehGas"].map({"'Regular'": 0, "'Diesel'": 1}),
            VehPower=lambda df: df[["VehPower"]].apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
            ),
            BonusMalus=lambda df: df["BonusMalus"] < 100,
        )
        .rename(columns=feature_name_map)
    )

    feature_names = data.columns

    frame = pd.concat([data, target], axis=1)

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()

    if return_X_y:
        return data, target

    french_motor_insurance_claims_dataset = Bunch(
        data=data,
        target=target,
        feature_names=np.array(feature_names),
        target_names=[target_names],
        frame=frame,
    )

    return french_motor_insurance_claims_dataset


def make_fake_data(
    y_fn: typing.Callable,
    n_samples: int,
    n_features: int,
    rng: np.random.Generator = np.random.default_rng(),
) -> pd.DataFrame:
    """
    This function generates fake data for testing purposes.

    Arguments
    ----------
    y_fn: callable
        A function that takes a 2D numpy array as input and returns a 1D numpy array.

    n_samples: int
        The number of samples to generate.

    n_features: int
        The number of features to generate.

    rng: np.random.Generator
        The random number generator to use.
        Defaults to :code:`np.random.default_rng()`.

    Returns
    -------
    X: pd.DataFrame
        The features.

    y: pd.Series
        The target.

    """
    X = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f"feature {i}" for i in range(n_features)],
    )

    y = pd.Series(y_fn(X.values), name="target")

    return X, y


def save_fake_data(
    X: pd.DataFrame, y: pd.Series, path: Path = Path.home().joinpath("data")
) -> None:
    """
    This function saves fake data to disk.
    This file will be saved in the directory
    specified by the path argument and
    called :code:`known_relationship.parquet`.

    Arguments
    ----------

    X: pd.DataFrame
        The features.

    y: pd.Series
        The target.

    path: str or Path
        The directory where the file should be saved.
        The file will be saved as :code:`known_relationship.parquet` in
        the specified directory.
        Defaults to :code:`Path.home().joinpath("data")`.

    Returns
    -------

    None

    """

    path = path.joinpath("known_relationship.parquet")

    X.join(y).to_parquet(path, index=False)

    return None


def load_fake_data(
    path: Path = Path.home().joinpath("data"),
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch:
    """
    Load and return the fake_data produced and saved using the
    functions :code:`make_fake_data` and :code:`save_fake_data`.

    Arguments
    ---------

    path: str or Path
        Path to the file where the data is stored or should be
        downloaded.
        Defaults to :code:`Path.home().joinpath("data")`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    try:
        fake_data_dataset = pd.read_parquet(path.joinpath("known_relationship.parquet"))
    except FileNotFoundError:
        raise ValueError(
            """
            The fake data has not been generated yet. 
            Please run the make_fake_data.py script.
            """
        )
    data = fake_data_dataset.drop(columns="target")
    target = fake_data_dataset["target"]

    feature_names = data.columns
    target_names = target.name

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()

    if return_X_y:
        return data, target

    fake_data_dataset = Bunch(
        data=data,
        target=target,
        feature_names=np.array(feature_names),
        frame=fake_data_dataset,
        target_names=np.array([target_names]),
    )

    return fake_data_dataset


def load_raw_dataset_frame(
    dataset_name: str = None,
    path: Path = Path.home().joinpath("data"),
) -> pd.DataFrame:
    """
    Load the raw dataset, without any of the preprocessing done.
    This includes the original feature names and target names.

    This is helpful for assessing whether the language model has
    previously seen these datasets.

    Arguments
    ---------

    dataset_name: str
        The name of the dataset to load.

    Returns
    -------

    pd.DataFrame
        The raw dataset.

    """

    if dataset_name == "uti":
        raise ValueError("The UTI dataset is not available in raw form.")

    if dataset_name == "fake_data":
        try:
            return pd.read_parquet(path.joinpath("known_relationship.parquet"))
        except FileNotFoundError:
            raise ValueError(
                """
                The fake data has not been generated yet. 
                Please run the make_fake_data.py script.
                """
            )

    elif dataset_name == "breast_cancer":
        return skd.load_breast_cancer(as_frame=True)["frame"]

    elif dataset_name == "california_housing":
        return skd.fetch_california_housing(as_frame=True, data_home=path)["frame"]

    elif dataset_name == "diabetes":
        return skd.fetch_openml(data_id=37, as_frame=True, data_home=path)["frame"]

    # all of the following are loaded from the UCI repository
    elif dataset_name in [
        "heart_disease",
        "wine_quality",
        "iris",
        "wine",
        "adult",
    ]:
        return load_and_save_uci_original_dataset(dataset_name=dataset_name, path=path)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
