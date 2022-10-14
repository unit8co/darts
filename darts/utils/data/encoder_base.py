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
from darts.logging import get_logger, raise_if
from darts.utils.timeseries_generation import generate_index

SupportedIndex = Union[pd.DatetimeIndex, pd.RangeIndex]
EncoderOutputType = Optional[Union[Sequence[TimeSeries], List[TimeSeries]]]
logger = get_logger(__name__)


class CovariateIndexGenerator(ABC):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
    ):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        """
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

    @abstractmethod
    def generate_train_series(
        self, target: TimeSeries, covariate: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:
        """
        Implement a method that extracts the required covariate index for training.

        Parameters
        ----------
        target
            The target TimeSeries used during training
        covariate
            Optionally, the future covariates used for training
        """
        pass

    @abstractmethod
    def generate_inference_series(
        self, n: int, target: TimeSeries, covariate: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:
        """
        Implement a method that extracts the required covariate index for prediction.

        Parameters
        ----------
        n
            The forecast horizon
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariate
            Optionally, the future covariates used for prediction
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


class PastCovariateIndexGenerator(CovariateIndexGenerator):
    """Generates index for past covariates on train and inference datasets"""

    def generate_train_series(
        self, target: TimeSeries, covariate: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:

        super().generate_train_series(target, covariate)
        target_end = target.end_time()
        return (
            covariate.time_index if covariate is not None else target.time_index,
            target_end,
        )

    def generate_inference_series(
        self, n: int, target: TimeSeries, covariate: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:
        """For prediction (`n` is given) with past covariates we have to distinguish between two cases:
        1)  If past covariates are given, we can use them as reference
        2)  If past covariates are missing, we need to generate a time index that starts `input_chunk_length`
            before the end of `target` and ends `max(0, n - output_chunk_length)` after the end of `target`
        """

        super().generate_inference_series(n, target, covariate)
        target_end = target.end_time()
        if covariate is not None:
            return covariate.time_index, target_end
        else:
            return (
                generate_index(
                    start=target.end_time()
                    - target.freq * (self.input_chunk_length - 1),
                    length=self.input_chunk_length
                    + max(0, n - self.output_chunk_length),
                    freq=target.freq,
                ),
                target_end,
            )

    @property
    def base_component_name(self) -> str:
        return "pc"


class FutureCovariateIndexGenerator(CovariateIndexGenerator):
    """Generates index for future covariates on train and inference datasets."""

    def generate_train_series(
        self, target: TimeSeries, covariate: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:
        """For training (when `n` is `None`) we can simply use the future covariates (if available) or target as
        reference to extract the time index.
        """

        super().generate_train_series(target, covariate)
        target_end = target.end_time()
        return (
            covariate.time_index if covariate is not None else target.time_index,
            target_end,
        )

    def generate_inference_series(
        self, n: int, target: TimeSeries, covariate: Optional[TimeSeries] = None
    ) -> Tuple[SupportedIndex, pd.Timestamp]:
        """For prediction (`n` is given) with future covariates we have to distinguish between two cases:
        1)  If future covariates are given, we can use them as reference
        2)  If future covariates are missing, we need to generate a time index that starts `input_chunk_length`
            before the end of `target` and ends `max(n, output_chunk_length)` after the end of `target`
        """
        super().generate_inference_series(n, target, covariate)
        target_end = target.end_time()
        if covariate is not None:
            return covariate.time_index, target_end
        else:
            return (
                generate_index(
                    start=target.end_time()
                    - target.freq * (self.input_chunk_length - 1),
                    length=self.input_chunk_length + max(n, self.output_chunk_length),
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
        covariate: Optional[TimeSeries] = None,
        merge_covariate: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Each subclass must implement a method to encode covariate index for training.

        Parameters
        ----------
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariate
            Optionally, the future covariates used for prediction
        merge_covariate
            Whether or not to merge the encoded TimeSeries with `covariate`.
        """
        pass

    @abstractmethod
    def encode_inference(
        self,
        n: int,
        target: TimeSeries,
        covariate: Optional[TimeSeries] = None,
        merge_covariate: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Each subclass must implement a method to encode covariate index for prediction

        Parameters
        ----------
        n
            The forecast horizon
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariate
            Optionally, the future covariates used for prediction
        merge_covariate
            Whether or not to merge the encoded TimeSeries with `covariate`.

        """
        pass

    @staticmethod
    def _merge_covariate(
        encoded: TimeSeries, covariate: Optional[TimeSeries] = None
    ) -> TimeSeries:
        """If (actual) covariates are given, merge the encoded index with the covariates

        Parameters
        ----------
        encoded
            The encoded TimeSeries either from `encode_train()` or `encode_inference()`
        covariate
            Optionally, the future covariates used for prediction
        """
        return covariate.stack(encoded) if covariate is not None else encoded

    @staticmethod
    def _drop_encoded_components(
        covariate: Optional[TimeSeries], components: pd.Index
    ) -> Optional[TimeSeries]:
        """Avoid pitfalls: `encode_train()` or `encode_inference()` can be called multiple times or chained.
        Exclude any encoded components from `covariate` to generate and add the new encodings at a later time.
        """
        if covariate is None:
            return covariate

        duplicate_components = components[components.isin(covariate.components)]
        # case 1: covariate only consists of encoded components
        if len(duplicate_components) == len(covariate.components):
            covariate = None
        # case 2: covariate has also non-encoded components
        elif len(duplicate_components) and len(duplicate_components) < len(
            covariate.components
        ):
            covariate = covariate[
                list(
                    covariate.components[
                        ~covariate.components.isin(duplicate_components)
                    ]
                )
            ]
        return covariate

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

    def __init__(self, index_generator: CovariateIndexGenerator):
        """Single encoders take an `index_generator` to generate the required index for encoding past and future
        covariates.
        See darts.utils.data.covariate_index_generators.py for the `CovariateIndexGenerator` subclasses.
        For past covariate encoders, use a `PastCovariateIndexGenerator`.
        For future covariate encoders use a `FutureCovariateIndexGenerator`.

        Parameters
        ----------
        index_generator
            An instance of `CovariateIndexGenerator` with methods `generate_train_series()` and
            `generate_inference_series()`. Used to generate the index for encoders.
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
        covariate: Optional[TimeSeries] = None,
        merge_covariate: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Returns encoded index for training.

        Parameters
        ----------
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariate
            Optionally, the covariate used for training: past covariate if `self.index_generator` is instance of
            `PastCovariateIndexGenerator`, future covariate if `self.index_generator` is instance of
            `FutureCovariateIndexGenerator`
        merge_covariate
            Whether or not to merge the encoded TimeSeries with `covariate`.
        """
        # exclude encoded components from covariate to add the newly encoded components later
        covariate = self._drop_encoded_components(covariate, self.components)

        # generate index and encodings
        index, target_end = self.index_generator.generate_train_series(
            target, covariate
        )
        encoded = self._encode(index, target_end, target.dtype)

        # optionally, merge encodings with original `covariate` series
        encoded = (
            self._merge_covariate(encoded, covariate=covariate)
            if merge_covariate
            else encoded
        )

        # save encoded component names
        if self.components.empty:
            components = encoded.components
            if covariate is not None:
                components = components[~components.isin(covariate.components)]
            self._components = components

        self._fit_called = True
        return encoded

    def encode_inference(
        self,
        n: int,
        target: TimeSeries,
        covariate: Optional[TimeSeries] = None,
        merge_covariate: bool = True,
        **kwargs,
    ) -> TimeSeries:
        """Returns encoded index for inference/prediction.

        Parameters
        ----------
        n
            The forecast horizon
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariate
            Optionally, the covariate used for prediction: past covariate if `self.index_generator` is instance of
            `PastCovariateIndexGenerator`, future covariate if `self.index_generator` is instance of
            `FutureCovariateIndexGenerator`
        merge_covariate
            Whether or not to merge the encoded TimeSeries with `covariate`.
        """
        # some encoders must be fit before `encode_inference()`
        raise_if(
            not self.fit_called and self.requires_fit,
            f"`{self.__class__.__name__}` object must be trained before inference. "
            f"Call method `encode_train()` before `encode_inference()`.",
            logger=logger,
        )

        # exclude encoded components from covariate to add the newly encoded components later
        covariate = self._drop_encoded_components(covariate, self.components)

        # generate index and encodings
        index, target_end = self.index_generator.generate_inference_series(
            n, target, covariate
        )
        encoded = self._encode(index, target_end, target.dtype)

        # optionally, merge encodings with original `covariate` series
        encoded = (
            self._merge_covariate(encoded, covariate=covariate)
            if merge_covariate
            else encoded
        )

        # optionally, save encoded component names also at inference as some encoders do not have to be trained before
        if self.components.empty:
            components = encoded.components
            if covariate is not None:
                components = components[~components.isin(covariate.components)]
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
        `"darts_enc_{covariate_temp}_{encoder}_{attribute}"`, where the elements are:

        * covariate_temp: "pc" or "fc" for past, or future covariates respectively.
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
    """`SequentialEncoderTransformer` applies transformation to the non-transformed encoded covariate output of
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

    def transform(self, covariate: List[TimeSeries]) -> List[TimeSeries]:
        """This method applies transformation to the non-transformed encoded covariate output of
        `SequentialEncoder._encode_sequence()` after being merged with user-defined covariates. The transformer is
        fitted when `transform()` is called for the first time. This ensures proper transformation of train, validation
        and inference dataset covariates. The masks ensure that no covariates are transformed that user explicitly
        supplied to `TorchForecastingModel.fit()` and `TorchForecastingModel.predict()`

        Parameters
        ----------
        covariate
            The non-transformed encoded covariate output of `SequentialEncoder._encode_sequence()` before merging with
            user-defined covariates.
        """
        if not self.fit_called:
            self._update_mask(covariate)
            if any(self.transform_mask):
                # fit the transformer on all encoded values by concatenating multi-series input encodings
                self.transformer.fit(
                    series=TimeSeries.from_values(
                        np.concatenate([cov.values() for cov in covariate]),
                        columns=covariate[0].components,
                    ),
                    component_mask=self.transform_mask,
                )
            self._fit_called = True

        if any(self.transform_mask):
            transformed = [
                self.transformer.transform(cov, component_mask=self.transform_mask)
                for cov in covariate
            ]
        else:
            transformed = covariate
        return transformed

    def _update_mask(self, covariate: List[TimeSeries]) -> None:
        """if user supplied additional covariates to model.fit() or model.predict(), `self.transform_mask` has to be
        updated as user-defined covariates should not be transformed. These covariates are always located in the
        first `n_diff = covariate[0].width - len(self.transform_mask)` components of each TimeSeries in in `covariate`.
        """

        n_diff = covariate[0].width - len(self.transform_mask)
        if not n_diff:
            pass
        else:
            self.transform_mask = np.array([False] * n_diff + list(self.transform_mask))

    @property
    def fit_called(self) -> bool:
        """Return whether or not the transformer has been fitted."""
        return self._fit_called
