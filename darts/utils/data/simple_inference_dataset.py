"""
Inference Dataset
-----------------
"""

from typing import Union, Sequence, Optional, Tuple

from .timeseries_dataset import TimeSeriesInferenceDataset
from ...timeseries import TimeSeries
from ...logging import raise_if_not


class SimpleInferenceDataset(TimeSeriesInferenceDataset):

    def __init__(self,
                 series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1):
        """
        Creates a dataset from lists of target series and corresponding covariate series and emits
        3-tuples of (tgt_past, cov_past, cov_future), all `TimeSeries` instances.
        `tgt_past` corresponds to the target series, which will be predicted into the future.
        `cov_past` is equal to the covariates with the same time index as the target series if covariates
        are provided, otherwise a value of `None` will be emitted.
        Both `tgt_past` and `cov_past` will be `input_chunk_length` long.
        If the model is required to produce the forecast over multiple iterations, i.e. if
        `n > self.output_chunk_length`, and if covariates were provided, then `cov_future` will
        be equal to the future covariates up to `n - self.output_chunk_length` time steps into the future.

        The parameter `input_chunk_length` is necessary to determine the minimum length for all `tgt_past`
        and `cov_past` time series.
        Parameters `n` and `output_chunk_length` are necessary to determine whether `cov_future` is necessary
        (or can be set to `None`) and to determine the required length for `cov_future`.
        It is important for the 3 emitted time series to have the same length across data points because
        they will be cast to tensors later on.

        Parameters
        ----------
        series
            The target series that are to be predicted into the future.
        covariates
            Optionally, the corresponding covariates that are used for predictions. This argument is required
            if the model was trained with covariates.
        n
            The number of time steps after the end of the training time series for which to produce predictions.
        input_chunk_length
            The length of the time series the model takes as input.
        output_chunk_length
            The length of the model predictions after one call to its `forward` function.
    """

        super().__init__()
        self.series = [series] if isinstance(series, TimeSeries) else series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates
        self.n = n
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        raise_if_not((covariates is None or len(series) == len(covariates)),
                     'The number of target series must be equal to the number of covariates.')

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx: int) -> Tuple[TimeSeries, Optional[TimeSeries], Optional[TimeSeries]]:

        target_series = self.series[idx]
        covariate_series = None if self.covariates is None else self.covariates[idx]

        raise_if_not(len(target_series) >= self.input_chunk_length,
                     'All input series must have length >= `input_chunk_length` ({}).'.format(
                     self.input_chunk_length))

        tgt_past = target_series[-self.input_chunk_length:]

        cov_past = cov_future = None
        if covariate_series is not None:

            # get first timestamp that lies in the future of target series
            first_pred_time = target_series.end_time() + target_series.freq

            # isolate past covariates and add them to array
            if covariate_series.end_time() >= first_pred_time:
                cov_past = covariate_series.drop_after(first_pred_time)
            else:
                cov_past = covariate_series
            cov_past = cov_past[-self.input_chunk_length:]

            # check whether future covariates are required
            if self.n > self.output_chunk_length:

                # check that enough future covariates are available
                last_required_future_covariate_ts = (
                    target_series.end_time() + (self.n - self.output_chunk_length) * target_series.freq
                )
                raise_if_not(covariate_series.end_time() >= last_required_future_covariate_ts,
                             'All covariates must be known `n - output_chunk_length` time steps into the future')

                # isolate necessary future covariates and add them to array
                cov_future = covariate_series.drop_before(first_pred_time - covariate_series.freq)
                cov_future = cov_future[:self.n - self.output_chunk_length]

        return tgt_past, cov_past, cov_future
