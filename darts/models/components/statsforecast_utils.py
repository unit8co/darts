import numpy as np

# In a normal distribution, 68.27 percentage of values lie within one standard deviation of the mean
one_sigma_rule = 68.27


def create_normal_samples(
    mu: float,
    std: float,
    num_samples: int,
    n: int,
) -> np.ndarray:
    """Generate samples assuming a Normal distribution."""
    samples = np.random.normal(loc=mu, scale=std, size=(num_samples, n)).T
    samples = np.expand_dims(samples, axis=1)
    return samples


def unpack_sf_dict(
    forecast_dict: dict,
):
    """Unpack the dictionary that is returned by the StatsForecast 'predict()' method."""
    mu = forecast_dict["mean"]
    std = forecast_dict[f"hi-{one_sigma_rule}"] - mu
    return mu, std
