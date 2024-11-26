import numpy as np
import llm_elicited_priors.datasets as datasets

SEED = 42
RNG = np.random.default_rng(SEED)

print("-" * 35)
print("Generating fake data with seed:", SEED)
print("-" * 35)


# define the function that generates the target variable
def y_fn(X):
    # y = 2 * x1 - 1 * x2 + 1 * x3 + noise
    true_y = 2 * X[:, 0] - 1 * X[:, 1] + X[:, 2]
    # some noise
    noise_y = 0.05 * RNG.normal(size=(X.shape[0]))
    return true_y + noise_y


# make fake data
X, y = datasets.make_fake_data(y_fn, n_samples=250, n_features=3, rng=RNG)

# save the fake data
datasets.save_fake_data(X, y)

# load the fake data
dataset = datasets.load_fake_data()

# print the some dataset information
print("X shape:", dataset["data"].shape)
print("y shape:", dataset["target"].shape)
print("Feature names:", dataset["feature_names"])
print("Target names:", dataset["target_names"])
