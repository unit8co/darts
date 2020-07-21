"""
Base Transformer
----------------
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Optional, List, Tuple

from darts.logging import raise_if_not, get_logger
from darts.preprocessing.validator import Validator

logger = get_logger(__name__)
T = TypeVar('T')


class BaseTransformer(Generic[T], ABC):
    def __init__(self,
                 validators: Optional[List[Validator]] = None,
                 validator_fns: Optional[List[Tuple[Callable[[T], bool], Optional[str]]]] = None,
                 reversible: bool = False,
                 can_predict: bool = False,
                 fittable: bool = False):
        """
        Abstract class for transformers. All deriving classes have to implement only one function `transform`.
        It also have `inverse_transform`, `fit` and `predict` left unimplemented. If child of this class implements
        any of this methods it should mark this with appropriate property (`reversible`, `fittable`, `can_predict`).

        Parameters
        ----------
        validators
            List of validators that will be called before transform
        validator_fns
            List of tuples of validating function and optional string with reason for validating.
            From this will be created list of validators and it will be appended to validators.
        reversible
            Flag indicating whether this transformer have implemented inverse_transform
        can_predict
            Flag indicating whether this transformer have implemented predict
        fittable
            Flag indicating whether this transformer have implemented fit
        """
        if validators:
            self._validators = validators
        else:
            self._validators = []
        if validator_fns:
            self._validators.extend((Validator(f, r) for f, r in validator_fns))

        self._reversible = reversible
        self._can_predict = can_predict
        self._fittable = fittable

    @property
    def reversible(self) -> bool:
        """
        Returns
        -------
        bool
            Whether transformer has inverse_transform.
        """
        return self._reversible

    @property
    def can_predict(self) -> bool:
        """
        Returns
        -------
        bool
            Whether transformer can be used to predict.
        """
        return self._can_predict

    @property
    def fittable(self) -> bool:
        """
        Returns
        -------
        bool
            Whether transformer can be fitted.
        """
        return self._fittable

    @property
    def name(self):
        """
        Returns
        -------
        str
            Name of transformer.
        """
        return "BaseTransformer"

    def validate(self, data: T) -> bool:
        """
        Validate data using validators set at init. If validation will be unsuccessful it will raise exception
        potentially with reason/explanation why validation failed.

        Parameters
        ----------
        data
            Object on which validation functions will be run.

        Returns
        -------
        True
            If successful.
        """
        for validator in self._validators:
            reason = validator.reason if validator.reason else ""
            raise_if_not(validator(data), f"Validation failed for {self.name}\n{reason}", logger)
        return True

    @abstractmethod
    def transform(self, data: T, *args, **kwargs) -> T:
        """
        Transform data. Not implemented in base class and has to be implemented by deriving classes.

        Parameters
        ----------
        data
            Object which will be transformed.
        args
            Positional arguments for the `transform` method
        kwargs
            Keyword arguments for the `transform` method

        Returns
        -------
        T
            Transformed data.
        """
        raise NotImplementedError(f"transform not implemented for transformer {self.name}")

    def inverse_transform(self, data: T, *args, **kwargs) -> T:
        """
        Inverse transformation of data. Not implemented in base class.

        Parameters
        ----------
        data
            Object which will be inverse transformed.
        args
            Positional arguments for the `inverse_transform` method
        kwargs
            Keyword arguments for the `inverse_transform` method

        Returns
        -------
        T
            Inverse transformed data.
        """
        raise NotImplementedError(f"inverse_transform not implemented for transformer {self.name}")

    def predict(self, data: T, *args, **kwargs) -> T:
        """
        Predict function. Not implemented in base class.

        Parameters
        ----------
        data
            Object on which predict will be performed.
        args
            Positional arguments for the `predict` method
        kwargs
            Keyword arguments for the `predict` method

        Returns
        -------
        T
            Predicted values.
        """
        raise NotImplementedError(f"predict not implemented for transformer {self.name}")

    def fit(self, data: T) -> 'BaseTransformer[T]':
        """
        Function which will fit transformer. Not implemented in base class.

        Parameters
        ----------
        data
            Object on which transformer will be fitter.

        Returns
        -------
        BaseTransformer[T]
            Fitted transformer.
        """
        raise NotImplementedError(f"fit not implemented for transformer {self.name}")

    def fit_transform(self, data: T, *args, **kwargs) -> T:
        """
        First fit transformer with data and then perform transformation on data.

        Parameters
        ----------
        data
            Object used to fit and transform.
        args
            Positional arguments for the `transform` method
        kwargs
            Keyword arguments for the `transform` method

        Returns
        -------
        T
            Transformed data.
        """
        return self.fit(data).transform(data, *args, **kwargs)

    def __call__(self, data: T, *args, fit: bool = False, inverse: bool = False, **kwargs) -> T:
        """
        Calling transformer will run validation and perform transformation or
        inverse_transformation on data. If fit flag is set transformer will first be fitted.

        Parameters
        ----------
        data
            On this validation, fit if applicable and transform will be run.
        fit
            If set transformer prior to validation will be fited.
        inverse
            if set inverse_transformation will be run instead of transform.
        args
            Positional arguments for the `transform` or `inverse_transform` method
        kwargs
            Keyword arguments for the `transform` or `inverse_transform` method

        Returns
        -------
        T
            Transformed data.
        """
        if fit:
            self.fit(data)

        self.validate(data)
        if inverse:
            return self.inverse_transform(data, *args, **kwargs)
        return self.transform(data, *args, **kwargs)
