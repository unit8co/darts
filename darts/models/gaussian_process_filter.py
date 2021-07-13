from typing import Optional

from darts.timeseries import TimeSeries
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from .filtering_model import FilteringModel
from ..utils.utils import raise_if_not


class GaussianProcessFilter(FilteringModel):
    def __init__(self,
                 kernel: Optional[Kernel] = None,
                 **kwargs):
        """ GaussianProcessFilter model
        This model uses the GaussianProcessRegressor of scikit-learn to fit a Gaussian Process to the
        supplied TimeSeries. This can then be used to obtain samples or the mean values of the
        Gaussian Process at the times of the TimeSeries.

        Parameters
        ----------
        kernel : sklearn.gaussian_process.kernels.Kernel, default: None
            The kernel specifying the covariance function of the Gaussian Process. If None is passed,
            the default in scikit-learn is used. Note that the kernel hyperparameters are optimized
            during fitting unless the bounds are marked as “fixed”.
        **kwargs
            Additional keyword arguments passed to `sklearn.gaussian_process.GaussianProcessRegressor`.
        """
        super().__init__()
        self.model = GaussianProcessRegressor(kernel=kernel, **kwargs)

    def filter(self,
               series: TimeSeries,
               num_samples: int = 1) -> TimeSeries:
        """
        Fits the Gaussian Process on the observations and returns samples from the Gaussian Process,
        or its mean values if `num_samples` is set to 1.

        Parameters
        ----------
        series : TimeSeries
            The series of observations used to infer the values according to the specified Gaussian Process.
            This must be a deterministic series (containing one sample).
        num_samples: int, default: 1
            Number of times a prediction is sampled from the Gaussian Process. If set to 1,
            instead the mean values will be returned.

        Returns
        -------
        TimeSeries
            A stochastic `TimeSeries` sampled from the Gaussian Process, or its mean
            if `num_samples` is set to 1.
        """
        raise_if_not(series.is_deterministic, 'The input series for the Gaussian Process filter must be '
                                              'deterministic (observations).')
        super().filter(series)

        times = series.time_index.values.reshape(-1, 1)
        values = series.values(copy=False)

        self.model.fit(times, values)

        if num_samples == 1:
            sampled_states = self.model.predict(times)
        else:
            sampled_states = self.model.sample_y(times, n_samples=num_samples)
        
        return TimeSeries.from_times_and_values(series.time_index, sampled_states)
