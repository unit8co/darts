from abc import ABC, abstractmethod
from ...timeseries import TimeSeries


class Encoder(ABC):
    def __init__(self, input_chunk_length, output_chunk_length):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

    @abstractmethod
    def generate_train_series(self,
                              target: TimeSeries,
                              covariates: TimeSeries) -> TimeSeries:
        pass

    @abstractmethod
    def generate_inference_series(self,
                                  target: TimeSeries,
                                  covariate: TimeSeries,
                                  n: int) -> TimeSeries:
        pass





class PastCovariatesEncoder(Encoder):
    pass


class FutureCovariatesEncoder(Encoder):
    def generate_train_series(self,
                              target: TimeSeries,
                              future_covariates: TimeSeries) -> TimeSeries:

        return 0
