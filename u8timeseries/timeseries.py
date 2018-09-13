from abc import ABCMeta, abstractmethod


class TimeseriesModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, train_set):
        pass

    def predict(self):
        pass