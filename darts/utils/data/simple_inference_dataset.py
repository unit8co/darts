"""
Inference Dataset
-----------------
"""

from typing import Union, Sequence, Optional, Tuple

from .timeseries_dataset import TimeSeriesInferenceDataset
from ...timeseries import TimeSeries
from ...logging import raise_if_not, raise_if


class SimpleInferenceDataset(TimeSeriesInferenceDataset):

    def __init__(self,
                 series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 model_is_recurrent: bool = False):
        """
        Creates a dataset from lists of target series and corresponding covariate series and emits
        3-tuples of (tgt_past, cov_past, cov_future), all `TimeSeries` instances.
        `tgt_past` corresponds to the target series, which will be predicted into the future.
        `cov_past` is equal to the covariates with the same time index as the target series if covariates
        are provided, otherwise a value of `None` will be emitted.
        Both `tgt_past` and `cov_past` will be `input_chunk_length` long.

        Block models:
        If the model is required to produce the forecast over multiple iterations, i.e. if
        `n > self.output_chunk_length`, and if covariates were provided, then `cov_future` will
        be equal to the future covariates up to `n - self.output_chunk_length` time steps into the future.

        Recurrent models:
        If covariates were used to train the model, `n` covariates have to be available into the future.
        Therefore, `cov_future` will be equal to the future covariates up to `n` time steps into the future.

        The parameter `input_chunk_length` is necessary to determine the minimum length for all `tgt_past`
        and `cov_past` time series.
        Parameters `n`, `output_chunk_length` and `model_is_recurrent` are necessary to determine whether
        `cov_future` is necessary (or can be set to `None`) and to determine the required length for `cov_future`.
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
        model_is_recurrent
            Boolean indicating whether the model that uses this dataset is recurrent or not.
    """

        super().__init__()
        self.series = [series] if isinstance(series, TimeSeries) else series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates
        self.n = n
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.model_is_recurrent = model_is_recurrent

        raise_if(model_is_recurrent and output_chunk_length != 1,
                 'Recurrent models require an `output_chunk_length == 1`.')

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
                cov_past = covariate_series.drop_after(first_pred_time + int(self.model_is_recurrent)
                                                       * covariate_series.freq)
            else:
                cov_past = covariate_series
            cov_past = cov_past[-self.input_chunk_length:]

            # check whether future covariates are required
            if self.n > self.output_chunk_length:

                # check that enough future covariates are available
                # block models need `n - output_chunk_length`
                # recurrent models need `n`
                last_required_future_covariate_ts = (
                    target_series.end_time() + (
                        self.n - self.output_chunk_length * (1 - int(self.model_is_recurrent)))
                    * target_series.freq
                )
                req_cov_string = 'n' if self.model_is_recurrent else 'n - output_chunk_length'
                raise_if_not(covariate_series.end_time() >= last_required_future_covariate_ts,
                             'All covariates must be known `{}` time steps into the future'.format(req_cov_string))

                # isolate necessary future covariates and add them to array
                cov_future = covariate_series.drop_before(first_pred_time - (1 - int(self.model_is_recurrent))
                                                          * covariate_series.freq)
                cov_future = cov_future[:self.n - self.output_chunk_length + int(self.model_is_recurrent)]
                """
                In the shifted dataset, for recurrent models, the covariates are shifted forward relative to the input
                series by one time step. This ensures that recurrent models have as input the most recent covariates
                when making a prediction (covariates with the same timestamp as the target).
                Because the RNN is trained that way, this shift has to be incorporated into `SimpleInferenceDataset`
                as well, and this applies to future covariates too. This translates to the different cutoff
                points seen at the creation of `cov_past` and `cov_future`.
                """

        return tgt_past, cov_past, cov_future
