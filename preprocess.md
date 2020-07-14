## Pipeline

```python
class Pipeline(Generic[T]):
    def __init__(self):
        pass
    
    # change from last time - passing args and kwargs to transformer
    def add_transformer(self, transformer: BaseTransformer[T], *transformer_args, **transformer_kwargs) -> 'Pipeline':
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
```

```python
ts # timeseries we are working with
transformers = [transformer_1, ..., transformer_n] # objects deriving from BaseTransformer
common_validators = [all_positive, ...]
```
## init pipeline either like that
```python
pipeline = Pipeline(transformers, common_validators)
```
## or like that
```python
pipeline = Pipeline()
for t in transformes:
    pipeline = pipeline.add_transformer(t)
for v in common_validators:
    pipeline = pipeline.add_validator(v)
```

## use like that

```python
pipeline.set_data(ts)
pipeline = pipeline.resolve_n_steps(1) # pipeline traversed using first trnasformation
ts_transformed = pipeline.get_data() # ts after 1st transform
pipeline = pipeline.resolve() # pipeline traversed to end (all transformations applied)
pipeline = pipeline.resolve(inverse=True) # pipeline reverted to begining
```

## But
Common pattern is first to split data, train/fit transformer on train part of data - apply all to train, test and val data.

So maybe instead of transformers and pipeline working on
`timeseries -> timeseries` making it `List[timeseries] -> List[timeseries]` would work better.


```Python

class BaseTransformer(Generic[T]):
    def __init__(self, validators):
        pass

    def add_validator(self, validator: Callable[[T], bool]):
        pass

    def validate(self, data: List[T]) -> bool:
        pass

    def transform(self, data: T, *args, **kwargs) -> List[T]:
        pass

    def inverse_transform(self, data: List[T], *args, **kwargs) -> List[T]:
        pass

    def fit(self, data: T):
        pass

    # inverse whether to apply inverse transformation
    # fit_on: index of data to fit on, -1 to skip fitting
    def __call__(self, data: List[T], *args, inverse: bool = False, fit_on: int = -1, **kwargs) -> List[T]:
        pass
```

```python
# Changes to Pipeline
class Pipeline(Generic[t])
    def set_data(data: List[T]) -> 'Pipeline':
        pass

    def get_data(self) -> List[T]:
        pass
```

then we could pass timeseries splitted in, next transformer could be fitted on train data, applied to all.
Example
```python
splitter = Splitter(by='2019-01-01') # splitter would also have to be able to join timeseires as inverse_transform
scaler = ScalerWrapperTransformer()
pipeline = (Pipeline()
    .add_tranformer(splitter)
    .add_transformer(scaler, fit_on=0)
    .set_data([ts]))

pipeline = pipeline.resolve()
train, val = pipeline.get_data()
original_ts = pipeline.resolve(reverse=True).get_data()
```

## Potential problems
What about transformers without inverse - maybe storing input and returning it would work but then we have to say that trasformer works only for one list of input.
Maybe overcomplicated