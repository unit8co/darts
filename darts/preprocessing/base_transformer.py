
from typing import TypeVar, Generic, Callable
from darts.logging import raise_if_not, get_logger

logger = get_logger(__name__)
T = TypeVar('T')


class BaseTransformer(Generic[T]):
    def __init__(self):
        self.validators = [
            lambda x: True
        ]

    def add_validator(self, validator: Callable[[T], bool]):
        self.validators.append(validator)
        return self

    def validate(self, data: T) -> bool:
        for validator in self.validators:
            raise_if_not(validator(data), "Validation failed", logger)
        return True

    def transform(self, data: T, *args, **kwargs) -> T:
        return data

    def inverse_transform(self, data: T, *args, **kwargs) -> T:
        return data

    def __call__(self, data: T, *args, **kwargs):
        self.validate(data)
        return self.transform(data, *args, **kwargs)
