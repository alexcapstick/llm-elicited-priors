from functools import partial
import jax
import jax.numpy as jnp
import blackjax
import numpy as np
import pymc as pm


def train_uninformative_logistic_regression(
    model,
    X_train,
    y_train,
    rng,
    n_samples: int,
    n_chains: int,
    mu: float = 0,
    sigma: float = 1,
):

    with model:
        theta = pm.Normal(
            "theta",
            mu=mu * np.ones(X_train.shape[1] + 1),
            sigma=sigma * np.ones(X_train.shape[1] + 1),
            shape=X_train.shape[1] + 1,
        )

        observations_data = pm.Data("observations", X_train, dims=("N", "D"))

        p = pm.Deterministic(
            "p", pm.math.invlogit(theta[0] + observations_data @ theta[1:]), dims=("N",)
        )

        outcomes = pm.Bernoulli("outcomes", p=p, observed=y_train, dims=("N",))

        idata = pm.sample(
            n_samples,
            tune=1000,
            chains=n_chains,
            return_inferencedata=True,
            random_seed=rng,
            cores=5,
            # nuts_sampler="blackjax",
            # nuts_sampler_kwargs=dict(
            #     chain_method="vectorized",
            # ),
        )

    return idata, model


def train_informative_logistic_regression(
    model,
    priors,
    X_train,
    y_train,
    rng,
    n_samples: int,
    n_chains: int,
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

        observations_data = pm.Data("observations", X_train, dims=("N", "D"))

        p = pm.Deterministic(
            "y", pm.math.invlogit(theta[0] + observations_data @ theta[1:]), dims=("N",)
        )

        outcomes = pm.Bernoulli("outcomes", p=p, observed=y_train, dims=("N",))

        idata = pm.sample(
            n_samples,
            tune=1000,
            chains=n_chains,
            return_inferencedata=True,
            random_seed=rng,
            cores=5,
            # nuts_sampler="blackjax",
            # nuts_sampler_kwargs=dict(
            #     chain_method="vectorized",
            # ),
        )

    return idata, model


def train_uninformative_linear_regression(
    model,
    X_train,
    y_train,
    rng,
    n_samples: int,
    n_chains: int,
    mu: float = 0,
    sigma: float = 1,
):
    with model:

        likelihood_sigma = pm.HalfCauchy(
            "sigma",
            beta=1,
        )
        theta = pm.Normal(
            "theta",
            mu * np.ones(X_train.shape[1] + 1),
            sigma=sigma * np.ones(X_train.shape[1] + 1),
        )

        observations_data = pm.Data("observations", X_train, dims=("N", "D"))

        likelihood = pm.Normal(
            "outcomes",
            mu=theta[0] + observations_data @ theta[1:],
            sigma=likelihood_sigma,
            observed=y_train,
            dims=("N",),
        )

        idata = pm.sample(
            n_samples,
            tune=1000,
            chains=n_chains,
            return_inferencedata=True,
            random_seed=rng,
            cores=5,
            # nuts_sampler="blackjax",
            # nuts_sampler_kwargs=dict(
            #     chain_method="vectorized",
            # ),
        )

        return idata, model


def train_informative_linear_regression(
    model,
    priors,
    X_train,
    y_train,
    rng,
    n_samples: int,
    n_chains: int,
):
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

        observations_data = pm.Data("observations", X_train, dims=("N", "D"))

        # Define likelihood
        likelihood = pm.Normal(
            "outcomes",
            mu=theta[0] + observations_data @ theta[1:],
            sigma=likelihood_sigma,
            observed=y_train,
            dims=("N",),
        )

        idata = pm.sample(
            n_samples,
            tune=1000,
            chains=n_chains,
            return_inferencedata=True,
            random_seed=rng,
            cores=5,
            # nuts_sampler="blackjax",
            # nuts_sampler_kwargs=dict(
            #     chain_method="vectorized",
            # ),
        )

        return idata, model


def predict_model(model, idata, X_test, rng):

    with model:
        pm.set_data(
            {
                "observations": X_test,
            }
        )
        posterior = pm.sample_posterior_predictive(
            idata,
            var_names=["outcomes"],
            predictions=True,
            random_seed=rng,
        )

    return posterior


def single_chain_inference(
    rng_key,
    initial_state,
    step_fn,
    num_samples: int,
):
    """

    Single chain inference using the step function
    on a set of parameters with shape (n_params,).

    Parameters
    ----------

    rng_key: jnp.ndarray
        The random key to use for the inference.

    initial_state: jnp.ndarray
        The initial state of the inference. This is returned
        using the init function of the random walk algorithm from
        :code:`blackjax`.

    step_fn: typing.Callable
        The step function for the inference. The function should take in a
        random key and the current state and return the next state and the
        acceptance probability. It should be taken from the :code:`blackjax`
        library.

    num_samples: int
        The number of samples to generate.

    Returns
    -------

    blackjax inference results
        The samples from the inference. The shape of the arrays should be
        (num_samples, n_params)


    """

    # single step inference to generate a new sample
    # from the current state
    @jax.jit
    def one_step(state, xs):
        _, rng_key = xs
        state, _ = step_fn(rng_key, state)
        # return the state and the state for the next iteration
        # since the first argument is the carry
        # and the second argument is stacked to the output
        return state, state

    # generate the keys for the inference
    keys = jax.random.split(rng_key, num_samples)

    # run the inference over the keys using a scan.
    # the first returned value is the final carry and
    # the second returned value is the stack of outputs
    scan_fn = blackjax.progress_bar.gen_scan_fn(num_samples, True)
    _, states = scan_fn(one_step, initial_state, (jnp.arange(num_samples), keys))

    return states


def multi_chain_inference(
    n_chains: int, rng_key, initial_states, step_fn, num_samples: int
):
    """

    Multi-chain inference using the step function
    on a set of parameters with shape (n_params,). We will
    vmap over the initial states and a random key will be generated
    for each chain.

    Parameters
    ----------

    n_chains: int
        The number of chains to run.

    rng_key: jnp.ndarray
        The random key to use for the inference.

    initial_state: jnp.ndarray
        The initial state of the inference. This is returned
        using the init function of the random walk algorithm from
        :code:`blackjax`. This shoule be of shape (n_chains, n_params).

    step_fn: typing.Callable
        The step function for the inference. The function should take in a
        random key and the current state and return the next state and the
        acceptance probability.

    num_samples: int
        The number of samples to generate.

    Returns
    -------

    blackjax inference results
        The samples from the inference. The shape of the arrays should be
        (n_chains, num_samples, n_params)

    """

    # vmap over the single chain inference.
    # we need to vmap over the rng_key and the initial_states
    # since we are running multiple chains.
    return jax.vmap(single_chain_inference, in_axes=(0, 0, None, None))(
        jax.random.split(rng_key, n_chains), initial_states, step_fn, num_samples
    )


def single_chain_with_adapt(
    rng_key,
    initial_point,
    logdensity_fn,
    algorithm,
    num_adapt_steps,
    num_samples,
    **algorithm_kwargs,
):
    """

    Single chain inference for a given algorithm with an
    adaptation phase to find the optimal step size and
    mass matrix for those that require it.

    Parameters
    ----------

    rng_key: jnp.ndarray
        The random key to use for the inference.

    initial_point: jnp.ndarray
        The initial points of the inference. This is
        used to initialise the adaptation phase.
        It should be of shape (n_params).

    algorithm: blackjax.algorithm
        The algorithm to use for the inference.

    num_adapt_steps: int
        The number of adaptation steps
        to take.

    num_samples: int
        The number of samples to generate.

    **algorithm_kwargs:
        The keyword arguments to pass to the algorithm. This
        is where the :code:`num_integration_steps` could be passed
        to the HMC algorithm.

    Returns
    -------

    blackjax inference results
        The samples from the inference. The shape of the arrays should be
        (n_chains, num_samples, n_params)

    """
    adapt = blackjax.window_adaptation(
        algorithm=algorithm,
        logdensity_fn=logdensity_fn,
        progress_bar=True,
        **algorithm_kwargs,
    )

    rng_key, adaption_key = jax.random.split(rng_key, 2)
    (initial_states, params), _ = adapt.run(
        adaption_key,
        initial_point,
        num_steps=num_adapt_steps,
    )

    sampler = algorithm(logdensity_fn, **params)

    rng_key, sample_key = jax.random.split(rng_key, 2)
    step_fn = jax.jit(sampler.step)

    return single_chain_inference(
        rng_key=sample_key,
        initial_state=initial_states,
        step_fn=step_fn,
        num_samples=num_samples,
    )


def multi_chain_inference_with_adapt(
    n_chains,
    rng_key,
    initial_points,
    logdensity_fn,
    algorithm,
    num_adapt_steps,
    num_samples,
    **algorithm_kwargs,
):
    """

    Multi-chain inference for a given algorithm with an
    adaptation phase to find the optimal step size and
    mass matrix for those that require it. We will
    vmap over the initial points and a random key will be generated
    for each chain.

    Parameters
    ----------

    n_chains: int
        The number of chains to run.

    rng_key: jnp.ndarray
        The random key to use for the inference.

    logdensity_fn: typing.Callable
        The log density function to use for the inference.

    initial_points: jnp.ndarray
        The initial points of the inference. This is
        used to initialise the adaptation phase.
        It should be of shape (n_chains, n_params).

    algorithm: blackjax.algorithm
        The algorithm to use for the inference.

    num_adapt_steps: int
        The number of adaptation steps
        to take.

    num_samples: int
        The number of samples to generate.

    **algorithm_kwargs:
        The keyword arguments to pass to the algorithm. This
        is where the :code:`num_integration_steps` could be passed
        to the HMC algorithm.

    Returns
    -------

    blackjax inference results
        The samples from the inference. The shape of the arrays should be
        (n_chains, num_samples, n_params)

    """

    single_chain_fn = partial(
        single_chain_with_adapt,
        logdensity_fn=logdensity_fn,
        algorithm=algorithm,
        num_adapt_steps=num_adapt_steps,
        num_samples=num_samples,
        **algorithm_kwargs,
    )

    return jax.vmap(single_chain_fn, in_axes=(0, 0))(
        jax.random.split(rng_key, n_chains), initial_points
    )


def sample_posterior_from_prior_samples(
    rng_key: jax.random.PRNGKey,
    prior_samples: jnp.ndarray,
    phi: jnp.ndarray,
    y: jnp.ndarray,
    algorithm,
    classification: bool,
    bw_method: float,
    n_chains: int,
    num_samples: int,
    num_adapt_steps: int,
):
    """
    This function takes prior samples and
    returns posterior samples using an adaption
    phase to find the optimal step size and mass
    matrix for the inference.


    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        The random key to use for the inference.

    prior_samples : jnp.ndarray
        The prior samples to use for the
        inference.

    phi : jnp.ndarray
        The design matrix.

    y : jnp.ndarray
        The target variable.

    algorithm: blackjax.algorithm
        The algorithm to use for the inference.

    classification : bool
        Whether the problem is a classification
        problem or not.

    bw_method : float
        The bandwidth method to use for the
        gaussian KDE.

    n_chains : int
        The number of chains to use.

    num_samples : int
        The number of samples to draw.

    num_adapt_steps : int
        The number of adaptation steps to use.

    Returns
    -------

    blackjax inference results
        The posterior samples.

    """

    _, M = phi.shape

    prior_kde = jax.scipy.stats.gaussian_kde(
        jnp.array(prior_samples.T), bw_method=bw_method
    )

    def logdensity_fn(w, classification=True):
        """

        The log-probability density function of
        the posterior distribution.

        This takes a single w of shape
        (num_params,) and returns a scalar.

        """

        if classification:
            p = jax.lax.logistic(phi @ w)

            # log likelihood using cross-entropy
            ll = lambda y, p: (y * jnp.log(p) + (1 - y) * jnp.log1p(-p))
            log_likelihood_term = ll(y, p).sum()
            prior_term = prior_kde.logpdf(w).sum()

        else:
            sigma = w[0]
            p = phi @ w[1:]

            # log likelihood using gaussian likelihood and prior
            # on the y error term
            half_cauchy_log_pdf = lambda x: (
                jnp.where(
                    x < 0,
                    -jnp.inf * jnp.ones_like(x),
                    jnp.log(2 / jnp.pi) - jnp.log1p(jnp.square(x)),
                )
            )

            ll = lambda y, p: jax.scipy.stats.norm.logpdf(y, loc=p, scale=sigma)

            log_likelihood_term = ll(y, p).sum()
            # prior sigma is just half cauchy with beta 1
            prior_term = (
                prior_kde.logpdf(w[1:]).sum() + half_cauchy_log_pdf(sigma).sum()
            )

        return prior_term + log_likelihood_term

    rng_key, initial_states_key = jax.random.split(rng_key, 2)
    initial_points = jax.random.multivariate_normal(
        initial_states_key,
        jnp.zeros((n_chains, M)),
        jnp.tile(jnp.eye(M)[None, :, :], (n_chains, 1, 1)),
    )

    if not classification:
        rng_key, initial_states_key = jax.random.split(rng_key, 2)
        initial_points = jnp.concatenate(
            [
                jnp.abs(jax.random.cauchy(initial_states_key, shape=(n_chains,)))[
                    :, None
                ],
                initial_points,
            ],
            axis=-1,
        )

    rng_key, sample_key = jax.random.split(rng_key, 2)
    states = multi_chain_inference_with_adapt(
        n_chains=n_chains,
        rng_key=sample_key,
        logdensity_fn=jax.jit(partial(logdensity_fn, classification=classification)),
        initial_points=initial_points,
        algorithm=algorithm,
        num_adapt_steps=num_adapt_steps,
        num_samples=num_samples,
    )

    return states
