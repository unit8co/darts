from abc import ABC, abstractmethod
from dateutils import relativedelta
import pandas as pd


class TimeseriesModel(ABC):
    """
    A simple interface to fit models (on some training set), and predict the n next data points
    """

    @abstractmethod
    def __init__(self):
        self.allowed_periodicity_str = {'second', 'minute', 'hour', 'day', 'week', 'month', 'year'}

        # Stores training date information (if provided):
        self.training_dates = None
        self.time_column = None
        self.periodicity_str = None

    @abstractmethod
    def fit(self, df, target_column, time_column, periodicity_str):
        """
        :param df: A Pandas DataFrame that contains at least one column with time series values
        :param target_column: the column containing time series values
        :param time_column: optionally, the column containing corresponding timestamps (can be None)
        :param periodicity_str: if [time_column] is provided, this must provide periodicty among following values:
               {'second', 'minute', 'hour', 'day', 'week', 'month', 'year'}

        TODO: Interpolate missing values
        """
        if time_column is not None:
            assert periodicity_str in self.allowed_periodicity_str, \
                   'periodicity_str argument must be in {}'.format(self.allowed_periodicity_str)

            self.time_column = time_column
            self.training_dates = df[time_column].values
            self.periodicity_str = periodicity_str

    @abstractmethod
    def predict(self, n):
        """
        :return: A Pandas DataFrame with following columns:
                 - y_hat: contains the point-predictions of the model
                 - [time_column] (optional): a time column (if a [time_column] is provided to fit())
                 - y_lower (optional): lower confidence bound
                 - y_upper (optional): upper confidence bound
        """
        pass

    def _get_new_dates(self, n):
        """
        This function creates a list of the n new dates (after the end of training set)
        :param n: number of dates after training set to generate
        """
        def _add_time_delta_to_datetime(np_dt, i):
            datet = pd.Timestamp(np_dt)

            switch = {
                'second': lambda dt: dt + relativedelta(seconds=+i),
                'minute': lambda dt: dt + relativedelta(minutes=+i),
                'hour': lambda dt: dt + relativedelta(hours=+i),
                'day': lambda dt: dt + relativedelta(days=+i),
                'week': lambda dt: dt + relativedelta(weeks=+i),
                'month': lambda dt: dt + relativedelta(months=+i),
                'year': lambda dt: dt + relativedelta(years=+i)
            }
            return switch[self.periodicity_str](datet)
        return [_add_time_delta_to_datetime(self.training_dates[-1], i) for i in range(1, n + 1)]

    def _build_forecast_df(self, point_preds, lower_bound=None, upper_bound=None):
        """
        Builds the pandas DataFrame to be returned by predict() method
        The column names are inspired from Prophet

        :param point_preds: a list or array of n point-predictions
        :param lower_bound: optionally, a list or array of lower bounds
        :param upper_bound:optionally, a list or array of upper bounds
        :return: a dataframe nicely formatted
        """

        columns = {
            'yhat': pd.Series(point_preds)
        }

        if self.time_column is not None:
            n = len(point_preds)
            new_dates = self._get_new_dates(n)
            columns[self.time_column] = pd.Series(new_dates)

        if lower_bound is not None:
            assert len(point_preds) == len(lower_bound), 'bounds should be same size as point predictions'
            columns['yhat_lower'] = lower_bound

        if upper_bound is not None:
            assert len(point_preds) == len(upper_bound), 'bounds should be same size as point predictions'
            columns['yhat_upper'] = upper_bound

        return pd.DataFrame(columns)


class SupervisedTimeSeriesModel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, df, target_column, feature_columns=None):
        """
        :param df: A Pandas DataFrame containing at least one feature column and one column with time series values (targets)
        :param feature_columns: A list of columns to use as features. If None, use all columns.
        """
        pass

    @abstractmethod
    def predict(self, df, feature_columns=None):
        """
        :param df: A Pandas DataFrame containing features columns
        :param feature_columns: A list of columns to use as features. If None, use all columns.
        :return: A Pandas Series containing the predictions for all rows in df
        """
        pass
