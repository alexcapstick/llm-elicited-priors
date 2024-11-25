import numpy as np
import jax.numpy as jnp
import jax
import tqdm


def log_bayes_factor_score(
    log_marginal_likelihood_1: float,
    log_marginal_likelihood_2: float,
) -> float:
    """
    Calculate the log bayes factor from two log
    marginal likelihoods.

    Arguments
    ----------
    log_marginal_likelihood_1 : float
        The log marginal likelihood of the first model.

    log_marginal_likelihood_2 : float
        The log marginal likelihood of the second model.

    Returns
    --------

    bayes_factor : float
        The bayes factor between the two models.

    """
    return log_marginal_likelihood_1 - log_marginal_likelihood_2


def bayes_factor_score(
    log_marginal_likelihood_1: float,
    log_marginal_likelihood_2: float,
) -> float:
    """
    Calculate the bayes factor from two log
    marginal likelihoods.

    If the log of the bayes factor is greater than 1e2, it will
    return infinity. This is to prevent numerical issues.

    Arguments
    ----------
    log_marginal_likelihood_1 : float
        The log marginal likelihood of the first model.

    log_marginal_likelihood_2 : float
        The log marginal likelihood of the second model.

    Returns
    --------

    bayes_factor : float
        The bayes factor between the two models.

    """

    log_bayes_factor_value = log_bayes_factor_score(
        log_marginal_likelihood_1,
        log_marginal_likelihood_2,
    )

    if log_bayes_factor_value > 1e2:
        return np.inf

    return np.exp(log_bayes_factor_value)


def negative_log_likelihood_jax(y: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """

    The negative log-likelihood of the model.

    Parameters
    ----------

    y: jnp.ndarray
        The true labels of the data. The
        shape of the array should be the same as
        the shape of the array :code:`p`.

    p: jnp.ndarray
        The predicted probabilities of the data. The
        shape of the array should be the same as
        the shape of the array :code:`y`.

    Returns
    -------

    jnp.ndarray
        The negative log-likelihood of the model. The shape of the array
        will be the same as the shape of the array :code:`y` and :code:`p`.

    """
    return -(y * jnp.log(p) + (1 - y) * jnp.log1p(-p))


def mean_pairwise_euclidean_distance(X, Y):
    """
    The mean pairwise Euclidean distance between two sets of points.

    Parameters
    ----------

    X: np.ndarray
        The first set of points. The shape of the array should be
        (n_samples, n_features).

    Y: np.ndarray
        The second set of points. The shape of the array should be
        (n_samples, n_features).


    Returns
    -------

    float
        The mean pairwise Euclidean distance between the two sets of points.


    """

    X, Y = jnp.array(X), jnp.array(Y)

    # Inner loop to compute pairwise distances between x and all y
    inner_loop = lambda x, Y: jax.vmap(
        lambda y: jnp.linalg.norm(x - y, ord=2), in_axes=0
    )(Y)
    jitted_inner_loop = jax.jit(inner_loop)

    # Outer loop to compute pairwise distances for all x
    outer_loop = jax.vmap(lambda x: jitted_inner_loop(x, Y), in_axes=0)(X)

    # Return the mean of all pairwise distances
    return jnp.mean(outer_loop)


def mean_pairwise_euclidean_distance_chunked(
    X,
    Y,
    chunk_size=100,
    verbose=False,
):
    """
    The mean pairwise Euclidean distance between two sets of points.
    This function computes the pairwise distances in chunks to reduce
    memory usage.


    Parameters
    ----------

    X: np.ndarray
        The first set of points. The shape of the array should be
        (n_samples, n_features).

    Y: np.ndarray
        The second set of points. The shape of the array should be
        (n_samples, n_features).


    Returns
    -------

    float
        The mean pairwise Euclidean distance between the two sets of points.

    """

    def compute_chunked_distances(x_chunk, y_chunk):
        # Use vmap to compute pairwise distances for the chunk
        # similar to mean_pairwise_euclidean_distance
        return jax.vmap(
            lambda x: jax.vmap(lambda y: jnp.linalg.norm(x - y, ord=2))(y_chunk)
        )(x_chunk)

    compute_chunked_distances = jax.jit(compute_chunked_distances)

    total_sum = 0.0

    X, Y = jnp.array(X), jnp.array(Y)

    # Iterate over chunks of X
    for i in tqdm.trange(0, X.shape[0], chunk_size, disable=not verbose):
        x_chunk = X[i : i + chunk_size]

        # Iterate over chunks of Y
        for j in range(0, Y.shape[0], chunk_size):
            y_chunk = Y[j : j + chunk_size]

            # Compute pairwise distances for the current chunk
            chunk_distances = compute_chunked_distances(x_chunk, y_chunk)

            # Sum the distances for this chunk
            total_sum += jnp.sum(chunk_distances) / Y.shape[0]

    return total_sum / X.shape[0]


def energy_distance(X, Y):
    """
    The energy distance between two sets of points.

    https://en.wikipedia.org/wiki/Energy_distance


    Parameters
    ----------

    X: np.ndarray
        The first set of points. The shape of the array should be
        (n_samples, n_features).

    Y: np.ndarray
        The second set of points. The shape of the array should be
        (n_samples, n_features).


    Returns
    -------

    float
        The energy distance between the two sets of points.

    """

    X, Y = jnp.array(X), jnp.array(Y)

    return (
        2 * mean_pairwise_euclidean_distance(X, Y)
        - mean_pairwise_euclidean_distance(X, X)
        - mean_pairwise_euclidean_distance(Y, Y)
    )


def energy_distance_chunked(X, Y, chunk_size=1000):
    """
    The energy distance between two sets of points.
    This function computes the pairwise distances in chunks to reduce
    memory usage.

    https://en.wikipedia.org/wiki/Energy_distance


    Parameters
    ----------

    X: np.ndarray
        The first set of points. The shape of the array should be
        (n_samples, n_features).

    Y: np.ndarray
        The second set of points. The shape of the array should be
        (n_samples, n_features).


    Returns
    -------

    float
        The energy distance between the two sets of points.

    """

    X, Y = jnp.array(X), jnp.array(Y)

    Exy = mean_pairwise_euclidean_distance_chunked(X, Y, chunk_size=chunk_size)
    Exx = mean_pairwise_euclidean_distance_chunked(X, X, chunk_size=chunk_size)
    Eyy = mean_pairwise_euclidean_distance_chunked(Y, Y, chunk_size=chunk_size)

    return jnp.sqrt(2 * Exy - Exx - Eyy)


def e_coefficient(X, Y, chunk_size=1000):
    """
    The e coefficient between two sets of points.
    This function computes the pairwise distances in chunks to reduce
    memory usage.

    https://en.wikipedia.org/wiki/Energy_distance


    Parameters
    ----------

    X: np.ndarray
        The first set of points. The shape of the array should be
        (n_samples, n_features).

    Y: np.ndarray
        The second set of points. The shape of the array should be
        (n_samples, n_features).


    Returns
    -------

    float
        The e-coefficient between the two sets of points.

    """

    Exy = mean_pairwise_euclidean_distance_chunked(X, Y, chunk_size=chunk_size)
    Exx = mean_pairwise_euclidean_distance_chunked(X, X, chunk_size=chunk_size)
    Eyy = mean_pairwise_euclidean_distance_chunked(Y, Y, chunk_size=chunk_size)

    return (2 * Exy - Exx - Eyy) / (2 * Exy)


def levenshtein_score(s1, s2, key=hash):
    """Levenshtein string distance with edits."""

    # Based on
    # https://github.com/interpretml/LLM-Tabular-Memorization-Checker/blob/d2252675a3ff9ac1d0887ca920d7d6fe347b3f66/tabmemcheck/utils.py#L614
    # Generate the cost matrix for the two strings
    def costmatrix(s1, s2, key=hash):
        rows = []

        previous_row = range(len(s2) + 1)
        rows.append(list(previous_row))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (key(c1) != key(c2))
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

            rows.append(previous_row)

        return rows

    rows = costmatrix(s1, s2, key)

    return rows[-1][-1]
