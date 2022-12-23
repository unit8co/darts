"""
Encoder Base Classes
--------------------
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import FittableDataTransformer
from darts.logging import get_logger, raise_if, raise_log
from darts.utils.timeseries_generation import generate_index

SupportedIndex = Union[pd.DatetimeIndex, pd.RangeIndex]
EncoderOutputType = Optional[Union[Sequence[TimeSeries], List[TimeSeries]]]
logger = get_logger(__name__)


class CovariatesIndexGenerator(ABC):
    def __init__(
        self,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_covariates: Optional[List[int]] = None,
    ):
        """:class:`CovariatesIndexGenerator` generates a time index for covariates at training and inference /
        prediction time with methods :func:`generate_train_idx()`, and :func:`generate_inference_idx()`.
        Without user `covariates`, it generates the minimum required covariate times spans for the corresponding
        scenarios described below. With user `covariates`, it simply copies and returns the `covariates` time index.

        It can be used:
        A   in combination with :class:`LocalForecastingModel`, or in a model agnostic scenario:
                All parameters can be ignored. This scenario is only supported by
                :class:`FutureCovariatesIndexGenerator`.
        B   in combination with :class:`RegressionModel`:
                Set `input_chunk_length`, `output_chunk_length`, and `lags_covariates`.
                `input_chunk_length` is the absolute value of the minimum target lag `abs(min(lags))` used with the
                regression model.
                Set `output_chunk_length`, and `lags_covariates` with the identical values used at forecasting model
                creation. For the covariates lags, use `lags_past_covariates` for class:`PastCovariatesIndexGenerator`,
                and `lags_future_covariates` for class:`PastCovariatesIndexGenerator`.
        C   in combination with :class:`TorchForecastingModel`:
                Set `input_chunk_length`, and `output_chunk_length` with the identical values used at forecasting model
                creation.

        Parameters
        ----------
        input_chunk_length
            Optionally, the number of input target time steps per chunk. Only required in scenarios B, C.
            Corresponds to `input_chunk_length` for :class:`TorchForecastingModel`, or to the absolute minimum target
            lag value `abs(min(lags))` for :class:`RegressionModel`.
        output_chunk_length
            Optionally, the number of output target time steps per chunk. Only required in scenarios B, and C.
            Corresponds to `output_chunk_length` for both :class:`TorchForecastingModel`, and :class:`RegressionModel`.
        lags_covariates
            Optionally, a list of integers giving the covariates lags used for Darts' RegressionModels. Only required
            in scenario B. Corresponds to the lag values from `lags_past_covariates` for past covariates, and
            `lags_future_covariates` for future covariates.
        """
        # check that parameters match one of the scenarios
        self._verify_scenario(input_chunk_length, output_chunk_length, lags_covariates)

        # input/output chunk length are guaranteed to both be `None`, or both be defined
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        # check lags validity
        min_covariates_lag = (
            min(lags_covariates) if lags_covariates is not None else None
        )
        max_covariates_lag = (
            max(lags_covariates) if lags_covariates is not None else None
        )
        self._verify_lags(min_covariates_lag, max_covariates_lag)

        # from verification min/max lags are guaranteed to either both be None, or both be an integer
        if min_covariates_lag is not None:
            # we add 1 to the lags so that shift == 0 represents the end of the target series (forecasting point)
            shift_start = min_covariates_lag + 1
            shift_end = max_covariates_lag + 1
        else:
            shift_start = None
            shift_end = None
        self.shift_start = shift_start
        self.shift_end = shift_end

    @abstractmethod
    def generate_train_idx(
        self, target: TimeSeries, covariates: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:
        """
        Generates/extracts time index (or integer index) for covariates at model training time.

        Parameters
        ----------
        target
            The target TimeSeries used during training.
        covariates
            Optionally, the covariates used for training.
            If given, the returned time index is equal to the `covariates` time index. Else, the returned time index
            covers the minimum required covariate time span for training a specific forecasting model. These
            requirements are derived from parameters set at :class:`CovariatesIndexGenerator` creation.
        """
        pass

    @abstractmethod
    def generate_inference_idx(
        self, n: int, target: TimeSeries, covariates: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:
        """
        Generates/extracts time index (or integer index) for covariates at model inference / prediction time.

        Parameters
        ----------
        n
            The forecasting horizon.
        target
            The target TimeSeries used during training or passed to prediction as `series`.
        covariates
            Optionally, the covariates used for prediction.
            If given, the returned time index is equal to the `covariates` time index. Else, the returned time index
            covers the minimum required covariate time spans for performing inference / prediction with a specific
            forecasting model. These requirements are derived from parameters set at :class:`CovariatesIndexGenerator`
            creation.
        """
        pass

    @property
    @abstractmethod
    def base_component_name(self) -> str:
        """Returns the index generator base component name.
        - "pc": past covariates
        - "fc": future covariates
        """
        pass

    def _verify_scenario(
        self,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_covariates: Optional[List[int]] = None,
    ):
        # LocalForecastingModel, or model agnostic (only supported by future covariates)
        is_scenario_a = (
            isinstance(self, FutureCovariatesIndexGenerator)
            and input_chunk_length is None
            and output_chunk_length is None
            and lags_covariates is None
        )
        # RegressionModel
        is_scenario_b = (
            input_chunk_length is not None
            and output_chunk_length is not None
            and lags_covariates is not None
        )
        # TorchForecastingModel
        is_scenario_c = (
            input_chunk_length is not None
            and output_chunk_length is not None
            and lags_covariates is None
        )

        if not any([is_scenario_a, is_scenario_b, is_scenario_c]):
            raise_log(
                ValueError(
                    "Invalid `CovariatesIndexGenerator` parameter combination: Could not be mapped to an existing "
                    "scenario, as defined in "
                    "https://unit8co.github.io/darts/generated_api/darts.dataprocessing.encoders.encoder_base.html"
                    "#darts.dataprocessing.encoders.encoder_base.CovariatesIndexGenerator"
                ),
                logger=logger,
            )

    def _verify_lags(self, min_covariates_lag, max_covariates_lag):
        """Check the base requirements for `min_covariates_lag` and `max_covariates_lag`:
        - both must either be None or an integer
        - min_covariates_lag < max_covariates_lag

        This method can be extended by subclasses for past and future covariates lag requirements.
        """
        # check that either None one of min/max_covariates_lag are given, or both are given
        if (min_covariates_lag is not None and max_covariates_lag is None) or (
            min_covariates_lag is None and max_covariates_lag is not None
        ):
            raise_log(
                ValueError(
                    "`min_covariates_lag` and `max_covariates_lag` must either both be `None` or both be integers"
                ),
                logger=logger,
            )
        if min_covariates_lag is not None:
            # check that if one of the two is given, both must be integers
            if not isinstance(min_covariates_lag, int) or not isinstance(
                max_covariates_lag, int
            ):
                raise_log(
                    ValueError(
                        "`min_covariates_lag` and `max_covariates_lag` must be both be integers."
                    ),
                    logger=logger,
                )
            # minimum lag must be less than maximum lag
            if min_covariates_lag > max_covariates_lag:
                raise_log(
                    ValueError(
                        "`min_covariates_lag` must be smaller than/equal to `max_covariates_lag`."
                    ),
                    logger=logger,
                )


class PastCovariatesIndexGenerator(CovariatesIndexGenerator):
    """Generates index for past covariates on train and inference datasets"""

    def generate_train_idx(
        self, target: TimeSeries, covariates: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:

        super().generate_train_idx(target, covariates)

        # the returned index depends on the following cases:
        # case 0
        #     user supplied covariates: simply return the covariate time index; guarantees that an exception is
        #     raised if user supplied insufficient covariates
        # case 1
        #     only input_chunk_length and output_chunk_length are given: the complete covariate index is within the
        #     target index; always True for all models except RegressionModels.
        # case 2
        #     covariate lags were given (shift_start <= 0 and shift_end <= 0) and
        #     abs(shift_start - 1) <= input_chunk_length: the complete covariate index is within the target index;
        #     can only be True for RegressionModels.
        # case 3
        #     covariate lags were given (shift_start <= 0 and shift_end <= 0) and
        #     abs(shift_start - 1) > input_chunk_length: we need to add indices before the beginning of the target
        #     series; can only be True for RegressionModels.

        target_end = target.end_time()
        if covariates is not None:  # case 0
            return covariates.time_index, target_end

        if self.shift_start is None:  # case 1
            steps_ahead_start = 0
        else:  # case 2 & 3
            steps_ahead_start = self.input_chunk_length + (self.shift_start - 1)

        if not self.shift_end:  # case 1
            steps_ahead_end = -self.output_chunk_length
        else:  # case 2 & 3
            steps_ahead_end = -(self.output_chunk_length - self.shift_end)

        steps_ahead_end = steps_ahead_end if steps_ahead_end else None
        return (
            _generate_train_idx(target, steps_ahead_start, steps_ahead_end),
            target_end,
        )

    def generate_inference_idx(
        self, n: int, target: TimeSeries, covariates: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:

        super().generate_inference_idx(n, target, covariates)

        # for prediction (`n` is given) with past covariates the returned index depends on the following cases:
        # case 0
        #     user supplied covariates: simply return the covariate time index; guarantees that an exception is
        #     raised if user supplied insufficient covariates.
        # case 1
        #     only input_chunk_length and output_chunk_length are given: we need to generate a time index that starts
        #     `input_chunk_length - 1` before the end of `target` and ends `max(0, n - output_chunk_length)` after the
        #     end of `target`; always True for all models except RegressionModels.
        # case 2
        #     covariate lags were given (shift_start <= 0 and shift_end <= 0): we need to generate a time index that
        #     starts `-shift_start` before the end of `target` and has a length of
        #     `shift_steps + max(0, n - output_chunk_length)`, where `shift_steps` is the number of time steps between
        #     `shift_start` and `shift_end`; can only be True for RegressionModels.

        target_end = target.end_time()
        if covariates is not None:  # case 0
            return covariates.time_index, target_end

        if self.shift_start is None or self.shift_end is None:  # case 1
            steps_back_end = self.input_chunk_length - 1
            n_steps = steps_back_end + 1 + max(0, n - self.output_chunk_length)
        else:  # case 2
            steps_back_end = -self.shift_start
            shift_steps = self.shift_end - self.shift_start + 1
            n_steps = shift_steps + max(0, n - self.output_chunk_length)

        return (
            generate_index(
                start=target.end_time() - target.freq * steps_back_end,
                length=n_steps,
                freq=target.freq,
            ),
            target_end,
        )

    @property
    def base_component_name(self) -> str:
        return "pc"

    def _verify_lags(self, min_covariates_lag, max_covariates_lag):
        # general lag checks
        super()._verify_lags(min_covariates_lag, max_covariates_lag)
        # check past covariate specific lag requirements
        if min_covariates_lag is not None and min_covariates_lag >= 0:
            raise_log(ValueError("`min_covariates_lag` must be < 0."), logger=logger)

        if max_covariates_lag is not None and max_covariates_lag >= 0:
            raise_log(ValueError("`max_covariates_lag` must be < 0."), logger=logger)


class FutureCovariatesIndexGenerator(CovariatesIndexGenerator):
    """Generates index for future covariates on train and inference datasets."""

    def generate_train_idx(
        self, target: TimeSeries, covariates: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:

        super().generate_train_idx(target, covariates)

        # the returned index depends on the following cases:
        # case 0
        #     user supplied covariates: simply return the covariate time index; guarantees that models raise an
        #     exception if user supplied insufficient covariates
        # case 1
        #     user uses a LocalForecastingModel or model agnostic scenario (input_chunk_length is None):
        #     simply return the target time index.
        # case 2
        #     only input_chunk_length and output_chunk_length are given: the complete covariate index is within the
        #     target index; always True for all models except RegressionModels.
        # case 3
        #     covariate lags were given and (shift_start <= 0 or shift_end <= 0): historic part of future covariates.
        #     if shift_end < there will only be the historic part of future covariates.
        #     If shift_start <= 0 and abs(shift_start - 1) > input_chunk_length: we need to add indices before the
        #     beginning of the target series; can only be True for RegressionModels.
        # case 4
        #     covariate lags were given and (shift_start > 0 or shift_end > 0): future part of future covariates.
        #     if shift_start > 0 there will only be the future part of future covariates.
        #     If shift_end > 0 and shift_start > input_chunk_length: we need to add indices after the end of the
        #     target series; can only be True for RegressionModels.

        target_end = target.end_time()

        if covariates is not None:  # case 0
            return covariates.time_index, target_end

        if self.input_chunk_length is None:  # case 1
            return target.time_index, target_end

        if self.shift_start is None:  # case 2
            steps_ahead_start = 0
        else:  # case 3
            steps_ahead_start = self.input_chunk_length + self.shift_start - 1

        if self.shift_end is None:  # case 2
            steps_ahead_end = 0
        else:  # case 4
            steps_ahead_end = -self.output_chunk_length + self.shift_end
        steps_ahead_end = steps_ahead_end if steps_ahead_end else None

        return (
            _generate_train_idx(target, steps_ahead_start, steps_ahead_end),
            target_end,
        )

    def generate_inference_idx(
        self, n: int, target: TimeSeries, covariates: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:

        super().generate_inference_idx(n, target, covariates)

        # for prediction (`n` is given) with future covariates the returned index depends on the following cases:
        # case 0
        #     user supplied covariates: simply return the covariate time index; guarantees that an exception is
        #     raised if user supplied insufficient covariates
        # case 1
        #     user uses a LocalForecastingModel or model agnostic scenario (input_chunk_length is None):
        #     simply return the target time index.
        # case 2
        #     only input_chunk_length and output_chunk_length are given: we need to generate a time index that starts
        #     `input_chunk_length - 1` before the end of `target` and ends `max(n, output_chunk_length)` after the
        #     end of `target`; always True for all models except RegressionModels.
        # case 3
        #     covariate lags were given: we need to generate a time index that starts `-shift_start`
        #     steps before the end of `target` and has a length of `shift_steps + max(0, n - output_chunk_length)`,
        #     where `shift_steps` is `shift_end - shift_start`; can only be True for RegressionModels.

        target_end = target.end_time()
        if covariates is not None:  # case 0
            return covariates.time_index, target_end

        if self.input_chunk_length is None:
            steps_back_end = -1
            n_steps = n
        elif self.shift_start is None:  # case 2
            steps_back_end = self.input_chunk_length - 1
            n_steps = steps_back_end + 1 + max(n, self.output_chunk_length)
        else:  # case 3
            steps_back_end = -self.shift_start
            shift_steps = self.shift_end + steps_back_end + 1
            n_steps = shift_steps + max(0, n - self.output_chunk_length)

        return (
            generate_index(
                start=target.end_time() - target.freq * steps_back_end,
                length=n_steps,
                freq=target.freq,
            ),
            target_end,
        )

    @property
    def base_component_name(self) -> str:
        return "fc"


class Encoder(ABC):
    """Abstract class for all encoders"""

    @abstractmethod
    def __init__(self):
        self.attribute = None
        self.dtype = np.float64
        self._fit_called = False

    @abstractmethod
    def encode_train(
        self,
        target: TimeSeries,
        covariates: Optional[TimeSeries] = None,
        merge_covariates: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Each subclass must implement a method to encode the covariates index for training.

        Parameters
        ----------
        target
            The target TimeSeries used during training or passed to prediction as `series`.
        covariates
            Optionally, the past or future covariates used for training.
        merge_covariates
            Whether or not to merge the encoded TimeSeries with `covariates`.
        """
        pass

    @abstractmethod
    def encode_inference(
        self,
        n: int,
        target: TimeSeries,
        covariates: Optional[TimeSeries] = None,
        merge_covariates: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Each subclass must implement a method to encode the covariates index for prediction.

        Parameters
        ----------
        n
            The forecast horizon
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariates
            Optionally, the past or future covariates used for prediction.
        merge_covariates
            Whether or not to merge the encoded TimeSeries with `covariates`.

        """
        pass

    @staticmethod
    def _merge_covariates(
        encoded: TimeSeries, covariates: Optional[TimeSeries] = None
    ) -> TimeSeries:
        """If (actual) covariates are given, merge the encoded index with the covariates

        Parameters
        ----------
        encoded
            The encoded TimeSeries either from `encode_train()` or `encode_inference()`
        covariates
            Optionally, some past or future covariates supplied by the user.
        """
        return covariates.stack(encoded) if covariates is not None else encoded

    @staticmethod
    def _drop_encoded_components(
        covariates: Optional[TimeSeries], components: pd.Index
    ) -> Optional[TimeSeries]:
        """Avoid pitfalls: `encode_train()` or `encode_inference()` can be called multiple times or chained.
        Exclude any encoded components from `covariates` to generate and add the new encodings at a later time.
        """
        if covariates is None:
            return covariates

        duplicate_components = components[components.isin(covariates.components)]
        # case 1: covariates only consist of encoded components
        if len(duplicate_components) == len(covariates.components):
            covariates = None
        # case 2: covariates also have non-encoded components
        elif len(duplicate_components) and len(duplicate_components) < len(
            covariates.components
        ):
            covariates = covariates[
                list(
                    covariates.components[
                        ~covariates.components.isin(duplicate_components)
                    ]
                )
            ]
        return covariates

    @property
    def fit_called(self) -> bool:
        """Returns whether the `Encoder` object has been fitted."""
        return self._fit_called

    @property
    @abstractmethod
    def requires_fit(self) -> bool:
        """Whether the `Encoder` sub class must be fit with `Encoder.encode_train()` before inference
        with `Encoder.encode_inference()`."""
        pass


class SingleEncoder(Encoder, ABC):
    """`SingleEncoder`: Abstract class for single index encoders.
    Single encoders can be used to implement new encoding techniques.
    Each single encoder must implement an `_encode()` method that carries the encoding logic.

    The `_encode()` method must take an `index` as input and generate a encoded single `TimeSeries` as output.
    """

    def __init__(self, index_generator: CovariatesIndexGenerator):
        """Single encoders take an `index_generator` to generate the required index for encoding past and future
        covariates.
        See darts.utils.data.covariate_index_generators.py for the `CovariatesIndexGenerator` subclasses.
        For past covariates encoders, use a `PastCovariatesIndexGenerator`.
        For future covariates encoders use a `FutureCovariatesIndexGenerator`.

        Parameters
        ----------
        index_generator
            An instance of `CovariatesIndexGenerator` with methods `generate_train_idx()` and
            `generate_inference_idx()`. Used to generate the index for encoders.
        """

        super().__init__()
        self.index_generator = index_generator
        self._components = pd.Index([])

    @abstractmethod
    def _encode(
        self, index: SupportedIndex, target_end: pd.Timestamp, dtype: np.dtype
    ) -> TimeSeries:
        """Single Encoders must implement an _encode() method to encode the index.

        Parameters
        ----------
        index
            The index generated from `self.index_generator` for either the train or inference dataset.
        target_end
            The end time of the target series.
        dtype
            The dtype of the encoded index
        """
        pass

    def encode_train(
        self,
        target: TimeSeries,
        covariates: Optional[TimeSeries] = None,
        merge_covariates: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Returns encoded index for training.

        Parameters
        ----------
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariates
            Optionally, the covariates used for training: past covariates if `self.index_generator` is a
            `PastCovariatesIndexGenerator`, future covariates if `self.index_generator` is a
            `FutureCovariatesIndexGenerator`
        merge_covariates
            Whether or not to merge the encoded TimeSeries with `covariates`.
        """
        # exclude encoded components from covariates to add the newly encoded components later
        covariates = self._drop_encoded_components(covariates, self.components)

        # generate index and encodings
        index, target_end = self.index_generator.generate_train_idx(target, covariates)
        encoded = self._encode(index, target_end, target.dtype)

        # optionally, merge encodings with original `covariates` series
        encoded = (
            self._merge_covariates(encoded, covariates=covariates)
            if merge_covariates
            else encoded
        )

        # save encoded component names
        if self.components.empty:
            components = encoded.components
            if covariates is not None:
                components = components[~components.isin(covariates.components)]
            self._components = components

        self._fit_called = True
        return encoded

    def encode_inference(
        self,
        n: int,
        target: TimeSeries,
        covariates: Optional[TimeSeries] = None,
        merge_covariates: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Returns encoded index for inference/prediction.

        Parameters
        ----------
        n
            The forecast horizon
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariates
            Optionally, the covariates used for prediction: past covariates if `self.index_generator` is a
            `PastCovariatesIndexGenerator`, future covariates if `self.index_generator` is a
            `FutureCovariatesIndexGenerator`
        merge_covariates
            Whether or not to merge the encoded TimeSeries with `covariates`.
        """
        # some encoders must be fit before `encode_inference()`
        raise_if(
            not self.fit_called and self.requires_fit,
            f"`{self.__class__.__name__}` object must be trained before inference. "
            f"Call method `encode_train()` before `encode_inference()`.",
            logger=logger,
        )

        # exclude encoded components from covariates to add the newly encoded components later
        covariates = self._drop_encoded_components(covariates, self.components)

        # generate index and encodings
        index, target_end = self.index_generator.generate_inference_idx(
            n, target, covariates
        )
        encoded = self._encode(index, target_end, target.dtype)

        # optionally, merge encodings with original `covariates` series
        encoded = (
            self._merge_covariates(encoded, covariates=covariates)
            if merge_covariates
            else encoded
        )

        # optionally, save encoded component names also at inference as some encoders do not have to be trained before
        if self.components.empty:
            components = encoded.components
            if covariates is not None:
                components = components[~components.isin(covariates.components)]
            self._components = components

        return encoded

    @property
    @abstractmethod
    def accept_transformer(self) -> List[bool]:
        """Whether the `SingleEncoder` sub class accepts to be transformed."""
        pass

    @property
    def components(self) -> pd.Index:
        """Returns the encoded component names. Only available after `Encoder.encode_train()` or
        `Encoder.encode_inference()` have been called."""
        return self._components

    @property
    @abstractmethod
    def base_component_name(self) -> str:
        """Returns the base encoder base component name. The string follows the given format:
        `"darts_enc_{covariates_temp}_{encoder}_{attribute}"`, where the elements are:

        * covariates_temp: "pc" or "fc" for past, or future covariates respectively.
        * encoder: the SingleEncoder type used:
            * "cyc" (cyclic temporal encoder),
            * "dta" (datetime attribute encoder),
            * "pos" (positional integer index encoder),
            * "cus" (custom callable index encoder)
        * attribute: the attribute used for the underlying encoder. Some examples:
            * "month_sin", "month_cos" (for "cyc")
            * "month" (for "dta")
            * "relative" (for "pos")
            * "custom" (for "cus")
        """
        return f"darts_enc_{self.index_generator.base_component_name}"


class SequentialEncoderTransformer:
    """`SequentialEncoderTransformer` applies transformation to the non-transformed encoded covariates output of
    `SequentialEncoder.encode_train()` and `SequentialEncoder.encode_inference()`. The transformer is fitted
    when `transform()` is called for the first time. This ensures proper transformation of train, validation and
    inference dataset covariates. User-supplied covariates are not transformed."""

    def __init__(
        self, transformer: FittableDataTransformer, transform_mask: List[bool]
    ):
        """
        Parameters
        ----------
        transformer
            A `FittableDataTransformer` object with a `fit_transform()` and `transform()` method.
        transform_mask
            A boolean 1-D mask specifying which of the input covariates to :meth:`transform()
            <SequentialEncoderTransformer.transform()>` must be transformed.
        """
        self.transformer: FittableDataTransformer = transformer
        self.transform_mask: np.ndarray = np.array(transform_mask)
        self._fit_called: bool = False

    def transform(self, covariates: List[TimeSeries]) -> List[TimeSeries]:
        """This method applies transformation to the non-transformed encoded covariates output of
        `SequentialEncoder._encode_sequence()` after being merged with user-defined covariates. The transformer is
        fitted when `transform()` is called for the first time. This ensures proper transformation of train, validation
        and inference dataset covariates. The masks ensure that no covariates are transformed that user explicitly
        supplied to `TorchForecastingModel.fit()` and `TorchForecastingModel.predict()`

        Parameters
        ----------
        covariates
            The non-transformed encoded covariates output of `SequentialEncoder._encode_sequence()` before merging with
            user-defined covariates.
        """
        if not self.fit_called:
            self._update_mask(covariates)
            if any(self.transform_mask):
                # fit the transformer on all encoded values by concatenating multi-series input encodings
                self.transformer.fit(
                    series=TimeSeries.from_values(
                        np.concatenate([cov.values() for cov in covariates]),
                        columns=covariates[0].components,
                    ),
                    component_mask=self.transform_mask,
                )
            self._fit_called = True

        if any(self.transform_mask):
            transformed = [
                self.transformer.transform(cov, component_mask=self.transform_mask)
                for cov in covariates
            ]
        else:
            transformed = covariates
        return transformed

    def _update_mask(self, covariates: List[TimeSeries]) -> None:
        """if user supplied additional covariates to model.fit() or model.predict(), `self.transform_mask` has to be
        updated as user-defined covariates should not be transformed. These covariates are always located in the
        first `n_diff = covariates[0].width - len(self.transform_mask)` components of each TimeSeries in in
        `covariates`.
        """

        n_diff = covariates[0].width - len(self.transform_mask)
        if not n_diff:
            pass
        else:
            self.transform_mask = np.array([False] * n_diff + list(self.transform_mask))

    @property
    def fit_called(self) -> bool:
        """Return whether or not the transformer has been fitted."""
        return self._fit_called


def _generate_train_idx(target, steps_ahead_start, steps_ahead_end) -> SupportedIndex:
    """The returned index depends on the following cases:

    case 1
        (steps_ahead_start >= 0 and steps_ahead_end is None or <= 1)
        the complete index is within the target index; always True for all models except RegressionModels.
    case 2
        steps_ahead_start < 0: add indices before the target start time; only possible for RegressionModels
        where the minimum past lag is larger than input_chunk_length.
    case 3
        steps_ahead_end > 0: add indices after the target end time; only possible for RegressionModels
        where the maximum future lag is larger than output_chunk_length.

    Parameters
    ----------
    target
        the target series.
    steps_ahead_start
        how many steps ahead of target start time to begin the index.
    steps_ahead_end
        how many steps ahead of target end time to end the index.
    """
    # case 1
    if steps_ahead_start >= 0 and (steps_ahead_end is None or steps_ahead_end <= -1):
        return target.time_index[steps_ahead_start:steps_ahead_end]

    # case 2
    idx_start = (
        generate_index(
            end=target.start_time() - target.freq,
            length=abs(steps_ahead_start),
            freq=target.freq,
        )
        if steps_ahead_start < 0
        else target.time_index.__class__([])
    )

    # if `steps_ahead_start >= 0` or `steps_ahead_end <= 0` we must extract a slice of the target series index
    center_start = None if steps_ahead_start < 0 else steps_ahead_start
    center_end = None if steps_ahead_end > 0 else steps_ahead_end
    idx_center = target.time_index[center_start:center_end]

    # case 3
    idx_end = (
        generate_index(
            start=target.end_time() + target.freq,
            length=abs(steps_ahead_end),
            freq=target.freq,
        )
        if steps_ahead_end > 0
        else target.time_index.__class__([])
    )

    # concatenate start, center, and end index
    # note: pandas' union() returns type pd.Index(), so we construct index directly from index class
    return target.time_index.__class__(idx_start.union(idx_center).union(idx_end))
