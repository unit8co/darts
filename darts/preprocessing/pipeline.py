
from typing import TypeVar, Generic
from darts.preprocessing.base_transformer import BaseTransformer

logger = get_logger(__name__)
T = TypeVar('T')


class Pipeline(Generic[T]):
    def __init__(self):
        pass

    def add_transformer(self, transformer: BaseTransformer[T]) -> 'Pipeline':
        pass

    # ? validator as separate class or as just callable ?
    def add_validator(self, validator) -> 'Pipeline':
        pass

    def resolve(self, inverse: bool = False) -> 'Pipeline':
        pass

    def resolve_n_steps(self, n: int) -> 'Pipeline':
        pass

    def set_data(data: T) -> 'Pipeline':
        pass

    def get_data(self) -> T:
        pass
